"""Feature engineering for coral heat-stress prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_heat_stress_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add SST anomaly and rolling heat-stress features."""
    features = frame.copy()
    sst = features["sst"]

    trailing_baseline = sst.rolling(window=30, min_periods=7).mean().shift(1)
    trailing_baseline = trailing_baseline.fillna(sst.expanding(min_periods=2).mean().shift(1))

    features["sst_baseline"] = trailing_baseline
    features["sst_anomaly"] = features["sst"] - features["sst_baseline"]
    features["hotspot"] = features["sst_anomaly"].clip(lower=0.0)
    features["rolling_heat_stress"] = features["hotspot"].rolling(window=84, min_periods=1).sum() / 7.0

    features["sst_lag_1"] = features["sst"].shift(1)
    features["sst_lag_7"] = features["sst"].shift(7)
    features["anomaly_lag_1"] = features["sst_anomaly"].shift(1)
    features["anomaly_trend_7"] = features["sst_anomaly"] - features["sst_anomaly"].shift(7)

    day_of_year = features["time"].dt.dayofyear.astype(float)
    features["season_sin"] = np.sin(2.0 * np.pi * day_of_year / 366.0)
    features["season_cos"] = np.cos(2.0 * np.pi * day_of_year / 366.0)
    return features


def add_future_risk_label(
    frame: pd.DataFrame,
    forecast_horizon: int,
    risk_delta: float,
) -> pd.DataFrame:
    """Label rows where rolling heat stress rises materially within the forecast window."""
    labeled = frame.copy()
    future_max_stress = (
        labeled["rolling_heat_stress"]
        .iloc[::-1]
        .rolling(window=forecast_horizon, min_periods=forecast_horizon)
        .max()
        .iloc[::-1]
    )
    labeled["future_risk"] = ((future_max_stress - labeled["rolling_heat_stress"]) >= risk_delta).astype("float")
    labeled.loc[future_max_stress.isna(), "future_risk"] = pd.NA
    return labeled


def feature_columns() -> list[str]:
    """Return the columns used by the model."""
    return [
        "sst",
        "sst_baseline",
        "sst_anomaly",
        "hotspot",
        "rolling_heat_stress",
        "sst_lag_1",
        "sst_lag_7",
        "anomaly_lag_1",
        "anomaly_trend_7",
        "season_sin",
        "season_cos",
    ]


def build_model_frame(
    frame: pd.DataFrame,
    forecast_horizon: int,
    risk_delta: float,
) -> pd.DataFrame:
    """Build the final training table used by the model."""
    engineered = add_heat_stress_features(frame)
    labeled = add_future_risk_label(
        engineered,
        forecast_horizon=forecast_horizon,
        risk_delta=risk_delta,
    )
    model_frame = labeled.dropna(subset=feature_columns() + ["future_risk"]).reset_index(drop=True)
    model_frame["future_risk"] = model_frame["future_risk"].astype(int)
    return model_frame
