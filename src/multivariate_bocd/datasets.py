"""Small datasets bundled with the package examples."""

from __future__ import annotations

from importlib import resources

import pandas as pd


def load_wamv_orientation(*, as_numpy: bool = True):
    """Load the bundled WAM-V roll/pitch orientation sample."""
    data_path = (
        resources.files("multivariate_bocd")
        / "data"
        / "time_filtered_ppangles_wamv_2024-01-23-09-51-25.csv"
    )
    frame = pd.read_csv(data_path)
    return frame.to_numpy() if as_numpy else frame
