from __future__ import annotations

from pathlib import Path

import numpy as np

from multivariate_bocd import BOCD, MultivariateGaussianWishartModel
from multivariate_bocd.plotting import plot_run_lengths


def generate_multivariate_data(
    *,
    n_segments: int = 4,
    segment_length: int = 120,
    dim: int = 3,
    seed: int = 11,
) -> tuple[np.ndarray, list[int]]:
    """Generate multivariate Gaussian data with known changepoints.

    Each segment has a different mean vector and a shared positive-definite
    covariance matrix. The returned changepoints are indices in the full data
    stream where a new segment begins.
    """
    rng = np.random.default_rng(seed)
    base_cov = 0.20 * np.eye(dim) + 0.05 * np.ones((dim, dim))
    means = rng.normal(loc=0.0, scale=2.5, size=(n_segments, dim))

    segments = [
        rng.multivariate_normal(mean=means[i], cov=base_cov, size=segment_length)
        for i in range(n_segments)
    ]
    data = np.vstack(segments)
    changepoints = [segment_length * i for i in range(1, n_segments)]
    return data, changepoints


def main() -> None:
    data, true_changepoints = generate_multivariate_data()

    prior_steps = 40
    prior = MultivariateGaussianWishartModel.init_from_data(
        data[:prior_steps],
        reset_prior_on_changepoint=True,
    )
    bocd = BOCD(model=prior, hazard=1 / 100)
    inference_data = data[prior_steps:]
    bocd.fit(inference_data)

    print(f"True changepoints in original stream: {true_changepoints}")
    print(f"Detector changepoints after prior window: {bocd.changepoints}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "synthetic_multivariate_bocd.png"
    plot_run_lengths(bocd, inference_data, show=False, save_path=str(output_path))
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
