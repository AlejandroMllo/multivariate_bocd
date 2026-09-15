"""Plotting helpers for tutorials and examples."""

from __future__ import annotations

import numpy as np


def plot_run_lengths(bocd, data, *, show: bool = True, save_path: str | None = None):
    """Plot observations and BOCD run-length probabilities."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install multivariate-bocd[examples] to use plotting helpers.") from exc

    run_probs = bocd.run_length_probabilities().detach().cpu().numpy()
    data = np.asarray(data)
    num_variables = data.shape[1]
    timesteps = np.arange(len(data))
    fig, axs = plt.subplots(num_variables + 1, 1, figsize=(10, max(4, num_variables + 2)))

    for i in range(num_variables):
        axs[i].plot(timesteps, data[:, i], lw=1.8, c="sienna")
        axs[i].vlines(
            x=bocd.changepoints,
            ymin=axs[i].get_ylim()[0],
            ymax=axs[i].get_ylim()[1],
            colors="gray",
            lw=1,
        )
        if i + 1 < num_variables:
            axs[i].tick_params(axis="x", bottom=False, top=False, labelbottom=False)

    run_probs = run_probs[max(0, run_probs.shape[0] - data.shape[0]) :, :]
    axs[-1].imshow(run_probs, aspect="auto", cmap="gray_r", norm=LogNorm(vmin=1e-4, vmax=1))
    axs[-1].set_xlim(axs[0].get_xlim())
    axs[-1].set_ylabel("Run length")
    axs[-1].set_xlabel("Timestep")
    fig.align_ylabels(axs)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axs
