"""Load NOAA SST data into a tidy pandas DataFrame."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


def load_sst_frame(path: Path) -> pd.DataFrame:
    """Load the NOAA NetCDF file and return a time-sorted SST frame."""
    dataset = xr.open_dataset(path)
    frame = (
        dataset[["analysed_sst"]]
        .to_dataframe()
        .reset_index()
        .rename(columns={"analysed_sst": "sst"})
        .sort_values("time")
        .reset_index(drop=True)
    )

    required_columns = {"time", "latitude", "longitude", "sst"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns in SST dataset: {missing_columns}")

    frame["time"] = pd.to_datetime(frame["time"], utc=False)
    return frame
