# Multivariate Bayesian Online Changepoint Detection

`multivariate-bocd` is a Python package for *Multivariate* [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742) (MBOCD). It is intended for streaming data where each observation is a vector and you want to update changepoint beliefs one sample at a time. The current predictive model uses a Gaussian-Wishart style prior for multivariate observations.

This implementation is derived from the multivariate BOCD module used in the IJRR paper [Situationally-Aware Dynamics Learning](https://doi.org/10.1177/02783649261431863) and its companion source repository, [AlejandroMllo/situationally_aware_dynamics_learning](https://github.com/AlejandroMllo/situationally_aware_dynamics_learning). If this repository is useful in your research, please cite the IJRR paper. Parts of this implementation are also inspired by Gregory Gundersen’s *univariate* Gaussian BOCD work[*](http://gregorygundersen.com/blog/2019/08/13/bocd/).

## Installation

For local development from a clone, use a virtual environment and install in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,examples]"
```

Editable mode links the installed package to this checkout, so edits under `src/multivariate_bocd` are picked up immediately. Inside a virtual environment this should not disturb your system Python. Remove it with `python -m pip uninstall multivariate-bocd`.

Install the package from [PyPI](https://pypi.org/project/multivariate-bocd/) with:

```bash
pip install multivariate-bocd
```

## Quick Start

```python
from multivariate_bocd import BOCD, MultivariateGaussianWishartModel, load_wamv_orientation

# Load bundled roll/pitch orientation sample data from the IJRR experiments.
data = load_wamv_orientation()[:1300]
data = 100.0 * data

prior_steps = 150
prior = MultivariateGaussianWishartModel.init_from_data(
    data[:prior_steps],
    reset_prior_on_changepoint=True,
)

bocd = BOCD(model=prior, hazard=1 / 250)
bocd.fit(data[prior_steps:])

print(bocd.changepoints)
print(bocd.run_length_probabilities().shape)
```

## Examples

Two runnable examples are included:

```bash
python examples/wamv_orientation.py
python examples/synthetic_multivariate.py
```

The WAM-V example uses the bundled roll/pitch orientation data. The synthetic example generates a multivariate Gaussian stream with known changepoints, which is useful for checking that the detector and plotting pipeline work on a clean toy problem.

## Repository Layout

```text
src/multivariate_bocd/      Package source
src/multivariate_bocd/data/ Bundled sample CSV data
examples/                  Runnable tutorials
tests/                     Smoke and API tests
```

## Notes on the Model

The detector follows the online recursion from Adams and MacKay (2007). The multivariate predictive model keeps a parameter history for each run-length hypothesis and can optionally refresh the prior when a changepoint is detected. This package keeps the implementation lightweight and focused on online use; for the complete situational-awareness robotics pipeline, see the companion IJRR repository linked above.

## Testing

From a fresh virtual environment:

```bash
python -m pip install -e ".[dev,examples]"
python -m pytest
python -m compileall -q src examples tests
```

Before publishing, build the wheel and install it in a separate clean environment. Maintainer publishing notes can live in an ignored local file such as `PUBLISHING_LOCAL.md`.

## Citation

Plain text:

> Alejandro Murillo-Gonzalez and Lantao Liu. "Situationally-Aware Dynamics Learning." The International Journal of Robotics Research. 2026. doi:10.1177/02783649261431863

BibTeX:

```bibtex
@article{murillo2026situationalawareness,
  author = {Alejandro Murillo-Gonzalez and Lantao Liu},
  title = {Situationally-Aware Dynamics Learning},
  journal = {The International Journal of Robotics Research},
  volume = {0},
  number = {0},
  pages = {02783649261431863},
  year = {2026},
  doi = {10.1177/02783649261431863},
  URL = {https://doi.org/10.1177/02783649261431863},
  eprint = {https://doi.org/10.1177/02783649261431863}
}
```

## License

MIT. See [`LICENSE`](LICENSE).
