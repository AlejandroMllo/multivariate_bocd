"""Multivariate Bayesian online changepoint detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

TensorLike = torch.Tensor | Iterable[float]


@dataclass(frozen=True)
class BOCDUpdate:
    """Result returned after adding one observation."""

    cp_prob: torch.Tensor
    grow_prob: torch.Tensor
    run_data: torch.Tensor


class BOCD:
    """Bayesian Online Changepoint Detection for streaming observations."""

    def __init__(self, model: "MultivariateGaussianWishartModel", hazard: float):
        if not 0.0 < hazard < 1.0:
            raise ValueError("hazard must be between 0 and 1.")

        self._model = model
        self._hazard = float(hazard)
        self._num_observations = 0
        self._log_H = torch.tensor(self._hazard, device=model.device, dtype=model.dtype).log()
        self._log_1mH = torch.tensor(1.0 - self._hazard, device=model.device, dtype=model.dtype).log()
        self._log_message = torch.zeros((), device=model.device, dtype=model.dtype)
        self._log_R = [self._log_message]
        self.changepoints: list[int] = []

    @property
    def num_observations(self) -> int:
        """Number of observations processed so far."""
        return self._num_observations

    @property
    def model(self) -> "MultivariateGaussianWishartModel":
        """Predictive model used by this detector."""
        return self._model

    def add_observation(self, obs: TensorLike) -> BOCDUpdate:
        """Process one observation and update the run-length distribution."""
        obs = self._model.as_tensor(obs)
        if obs.ndim != 1:
            raise ValueError("obs must be a one-dimensional observation vector.")
        if obs.shape[0] != self._model.d:
            raise ValueError(f"obs has dimension {obs.shape[0]}, expected {self._model.d}.")

        self._num_observations += 1
        log_pis = self._model.log_pred_prob(self._num_observations, obs)
        log_growth_probs = log_pis + self._log_message + self._log_1mH
        log_cp_prob = torch.logsumexp(log_pis + self._log_message + self._log_H, dim=0)
        new_log_joint = torch.cat([log_cp_prob.unsqueeze(dim=0), log_growth_probs], dim=0)
        evidence = torch.logsumexp(new_log_joint, dim=0)
        log_run_length_prob = new_log_joint - evidence
        self._log_R.append(log_run_length_prob)
        self._model.update_params(self._num_observations, obs)

        if log_run_length_prob[0] > log_run_length_prob[-1]:
            changepoint = self._num_observations
            self.changepoints.append(changepoint)
            self._model.register_changepoint(changepoint)
            self._log_message = torch.zeros((), device=self._model.device, dtype=self._model.dtype)
        else:
            self._log_message = new_log_joint

        return BOCDUpdate(
            cp_prob=log_run_length_prob[0].exp(),
            grow_prob=log_run_length_prob[-1].exp(),
            run_data=self._model.current_run_data(),
        )

    def fit(self, data: TensorLike) -> "BOCD":
        """Process a sequence of observations and return ``self``."""
        data = self._model.as_tensor(data)
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array.")
        for obs in data:
            self.add_observation(obs)
        return self

    def run_length_probabilities(self) -> torch.Tensor:
        """Return a dense matrix of run-length probabilities."""
        length = len(self._log_R)
        run_probs = torch.full(
            (length, length), -torch.inf, device=self._model.device, dtype=self._model.dtype
        )
        for timestep, timestep_probs in enumerate(self._log_R):
            run_length = len(timestep_probs) if timestep_probs.ndim > 0 else 1
            run_probs[-run_length:, timestep] = timestep_probs
        return run_probs.exp()


class MultivariateGaussianWishartModel:
    """Gaussian-Wishart style predictive model for BOCD."""

    def __init__(
        self,
        mu0: TensorLike,
        kappa0: float | torch.Tensor,
        nu0: float | torch.Tensor,
        T0: TensorLike,
        reset_prior_on_changepoint: bool = False,
        *,
        regularization: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self.mu0 = self.as_tensor(mu0)
        self.device = self.mu0.device
        self.T0 = self.as_tensor(T0)
        self.kappa0 = self.as_tensor(kappa0)
        self.nu0 = self.as_tensor(nu0)

        if self.mu0.ndim != 1:
            raise ValueError("mu0 must be one-dimensional.")
        if self.T0.shape != (self.mu0.shape[0], self.mu0.shape[0]):
            raise ValueError("T0 must have shape (d, d).")
        if self.kappa0 <= 0:
            raise ValueError("kappa0 must be positive.")
        if self.nu0 <= self.mu0.shape[0] - 1:
            raise ValueError("nu0 must be greater than d - 1.")
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")

        self.d = self.mu0.shape[0]
        self.regularization = float(regularization)
        self._reset_prior_on_changepoint = bool(reset_prior_on_changepoint)
        self.data = torch.empty((0, self.d), device=self.device, dtype=self.dtype)
        self.mu_history = self.mu0.clone().unsqueeze(dim=0)
        self.kappa_history = self.kappa0.clone().reshape(1)
        self.nu_history = self.nu0.clone().reshape(1)
        self.T_history = self.T0.clone().unsqueeze(dim=0)
        self._changepoints = [0]

    def as_tensor(self, value: TensorLike) -> torch.Tensor:
        """Convert values to this model's dtype and device."""
        if torch.is_tensor(value):
            tensor = value
            if self.device is not None:
                tensor = tensor.to(device=self.device)
            return tensor.to(dtype=self.dtype)
        return torch.as_tensor(value, device=self.device, dtype=self.dtype)

    def register_changepoint(self, timestep: int) -> None:
        """Register a detected changepoint and optionally refresh the prior."""
        if timestep < 0:
            raise ValueError("timestep must be non-negative.")

        if self._reset_prior_on_changepoint:
            latest = self.latest_changepoint()
            run_data = self.data[latest:]
            if len(run_data) > 0:
                self.mu0 = run_data.mean(dim=0)
                self.kappa0 = torch.tensor(float(run_data.shape[0]), device=self.device, dtype=self.dtype)
                self.nu0 = torch.tensor(float(self.d), device=self.device, dtype=self.dtype)
                if run_data.shape[0] > self.d:
                    self.T0 = self._precision_from_data(run_data, fallback=self.T0)
        self._changepoints.append(int(timestep))

    def current_run_data(self) -> torch.Tensor:
        """Observations seen since the latest registered changepoint."""
        return self.data[self.latest_changepoint() :]

    def latest_changepoint(self) -> int:
        """Most recent changepoint index, using the detector's time base."""
        return self._changepoints[-1]

    def latest_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the latest ``mu``, ``kappa``, ``nu``, and ``T`` values."""
        return self.mu_history[-1], self.kappa_history[-1], self.nu_history[-1], self.T_history[-1]

    def changepoints(self) -> list[int]:
        """Return changepoints registered by the model."""
        return list(self._changepoints)

    def log_pred_prob(self, t: int, x: TensorLike) -> torch.Tensor:
        """Compute log predictive probabilities for each run-length hypothesis."""
        x = self.as_tensor(x)
        t0 = self.latest_changepoint()
        means = self.mu_history[t0:t]
        precisions = self._regularized_precision(self.T_history[t0:t])
        diffs = x - means
        quad_form = torch.sum(
            diffs * torch.matmul(precisions, diffs.unsqueeze(-1)).squeeze(-1), dim=-1
        )
        _, logdet_precision = torch.linalg.slogdet(precisions)
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=self.dtype))
        return -0.5 * quad_form + 0.5 * logdet_precision - 0.5 * self.d * log_2pi

    def update_params(self, t: int, x: TensorLike) -> None:
        """Update sufficient-statistic histories after observing ``x``."""
        del t
        x = self.as_tensor(x)
        self.data = torch.cat([self.data, x.unsqueeze(dim=0)], dim=0)
        t0 = self.latest_changepoint()
        n = torch.tensor(1.0, device=self.device, dtype=self.dtype)
        data_mean = x

        mu_n = (
            self.kappa_history.unsqueeze(1) * self.mu0 + n * data_mean
        ) / (self.kappa_history + n).unsqueeze(1)
        self.mu_history = torch.cat([self.mu0.unsqueeze(dim=0), mu_n], dim=0)

        data_diff = self.data[t0:, :] - data_mean
        S = torch.matmul(data_diff.T, data_diff)
        mean_diff = (self.mu0 - data_mean).reshape(-1, 1)
        mean_diff_cov = torch.matmul(mean_diff, mean_diff.T)
        mean_diff_scale = ((self.kappa_history * n) / (self.kappa_history + n)).view(-1, 1, 1)
        T_n = self.T0 + S + mean_diff_scale * mean_diff_cov
        self.T_history = torch.cat([self.T0.unsqueeze(dim=0), T_n], dim=0)
        self.nu_history = torch.cat([self.nu0.unsqueeze(0), self.nu_history + n], dim=0)
        self.kappa_history = torch.cat([self.kappa0.unsqueeze(0), self.kappa_history + n], dim=0)

    @classmethod
    def init_from_data(
        cls,
        data: TensorLike,
        reset_prior_on_changepoint: bool = False,
        *,
        regularization: float = 1e-6,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> "MultivariateGaussianWishartModel":
        """Initialize prior parameters from a matrix of observations."""
        data = torch.as_tensor(data, device=device, dtype=dtype)
        if data.ndim != 2:
            raise ValueError("data must be a two-dimensional array.")
        if data.shape[0] < 2:
            raise ValueError("at least two observations are required to initialize from data.")

        mean = torch.mean(data, dim=0)
        precision = cls._precision_from_data_static(data, regularization=regularization)
        return cls(
            mu0=mean,
            kappa0=float(data.shape[0]),
            nu0=float(mean.shape[0]),
            T0=precision,
            reset_prior_on_changepoint=reset_prior_on_changepoint,
            regularization=regularization,
            device=data.device,
            dtype=dtype,
        )

    def _regularized_precision(self, precision: torch.Tensor) -> torch.Tensor:
        if self.regularization == 0:
            return precision
        eye = torch.eye(self.d, device=self.device, dtype=self.dtype)
        return precision + self.regularization * eye.unsqueeze(0)

    def _precision_from_data(self, data: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        try:
            return self._precision_from_data_static(data, regularization=self.regularization)
        except RuntimeError:
            return fallback

    @staticmethod
    def _precision_from_data_static(data: torch.Tensor, regularization: float) -> torch.Tensor:
        covariance = torch.cov(data.T)
        eye = torch.eye(covariance.shape[0], device=data.device, dtype=data.dtype)
        covariance = covariance + regularization * eye
        return torch.linalg.pinv(covariance)
