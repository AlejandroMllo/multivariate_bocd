import numpy as np
import torch

from multivariate_bocd import BOCD, MultivariateGaussianWishartModel, load_wamv_orientation


def test_load_bundled_sample_data():
    data = load_wamv_orientation()
    assert data.shape[1] == 4
    assert data.shape[0] > 100


def test_bocd_fit_returns_run_length_probabilities():
    rng = np.random.default_rng(7)
    prior_data = rng.normal(size=(20, 3))
    stream = rng.normal(size=(12, 3))

    model = MultivariateGaussianWishartModel.init_from_data(prior_data)
    detector = BOCD(model=model, hazard=0.05).fit(stream)
    run_probs = detector.run_length_probabilities()

    assert detector.num_observations == len(stream)
    assert run_probs.shape == (len(stream) + 1, len(stream) + 1)
    assert torch.isfinite(run_probs).all()
    assert torch.allclose(run_probs.sum(dim=0), torch.ones(run_probs.shape[1], dtype=run_probs.dtype))


def test_invalid_observation_dimension_raises():
    model = MultivariateGaussianWishartModel.init_from_data(np.ones((4, 2)))
    detector = BOCD(model=model, hazard=0.1)

    try:
        detector.add_observation(np.ones(3))
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("expected dimension mismatch to raise ValueError")


def test_synthetic_example_generator_runs_detector():
    from examples.synthetic_multivariate import generate_multivariate_data

    data, changepoints = generate_multivariate_data(n_segments=3, segment_length=30, dim=2, seed=5)
    assert data.shape == (90, 2)
    assert changepoints == [30, 60]

    model = MultivariateGaussianWishartModel.init_from_data(data[:20])
    detector = BOCD(model=model, hazard=0.1).fit(data[20:30])
    assert detector.run_length_probabilities().shape == (11, 11)
