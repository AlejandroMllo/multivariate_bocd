"""Public API for multivariate Bayesian online changepoint detection."""

from .datasets import load_wamv_orientation
from .mbocd import BOCD, BOCDUpdate, MultivariateGaussianWishartModel

__all__ = [
    "BOCD",
    "BOCDUpdate",
    "MultivariateGaussianWishartModel",
    "load_wamv_orientation",
]

__version__ = "0.1.0"
