from __future__ import annotations

from pathlib import Path

from multivariate_bocd import BOCD, MultivariateGaussianWishartModel, load_wamv_orientation
from multivariate_bocd.plotting import plot_run_lengths


def main() -> None:
    data = load_wamv_orientation()[:1300]
    data = 100.0 * data

    prior_steps = 150
    prior = MultivariateGaussianWishartModel.init_from_data(
        data[:prior_steps],
        reset_prior_on_changepoint=True,
    )
    bocd = BOCD(model=prior, hazard=1 / 250)
    inference_data = data[prior_steps:]
    bocd.fit(inference_data)

    print(f"Processed {bocd.num_observations} observations")
    print(f"Detected changepoints: {bocd.changepoints}")

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    plot_run_lengths(bocd, inference_data, show=False, save_path=str(output_dir / "wamv_bocd.png"))
    print(f"Saved plot to {output_dir / 'wamv_bocd.png'}")


if __name__ == "__main__":
    main()
