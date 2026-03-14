#!/usr/bin/env python3
"""Train the coral heat-stress model from NOAA SST data."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .data_loader import load_sst_frame
    from .features import build_model_frame, feature_columns
    from .model import evaluate_model, split_train_test, train_xgboost
except ImportError:
    from data_loader import load_sst_frame
    from features import build_model_frame, feature_columns
    from model import evaluate_model, split_train_test, train_xgboost


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training pipeline."""
    parser = argparse.ArgumentParser(description="Train the coral heat-stress model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/raw/noaaSSTcoralData.nc"),
        help="Path to the NOAA SST NetCDF dataset",
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=14,
        help="Days ahead used to define the future risk label",
    )
    parser.add_argument(
        "--risk-delta",
        type=float,
        default=0.75,
        help="Minimum future increase in rolling heat stress used for the risk label",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for model reproducibility",
    )
    return parser.parse_args()


def run_training(
    data_path: Path,
    forecast_horizon: int,
    risk_delta: float,
    random_state: int,
) -> dict[str, object]:
    """Run the full training and evaluation pipeline."""
    raw_frame = load_sst_frame(data_path)
    model_frame = build_model_frame(
        raw_frame,
        forecast_horizon=forecast_horizon,
        risk_delta=risk_delta,
    )
    train_frame, test_frame = split_train_test(model_frame)
    model = train_xgboost(train_frame, random_state=random_state)
    metrics = evaluate_model(model, test_frame)
    metrics.update(
        {
            "rows_loaded": len(raw_frame),
            "rows_modeled": len(model_frame),
            "feature_columns": feature_columns(),
            "train_positive_rate": train_frame["future_risk"].mean(),
            "train_start": train_frame["time"].min(),
            "train_end": train_frame["time"].max(),
        }
    )
    return metrics


def main() -> int:
    """Entry point for CLI training."""
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    if args.forecast_horizon <= 0:
        raise ValueError("--forecast-horizon must be positive")
    if args.risk_delta <= 0:
        raise ValueError("--risk-delta must be positive")

    metrics = run_training(
        data_path=args.data,
        forecast_horizon=args.forecast_horizon,
        risk_delta=args.risk_delta,
        random_state=args.random_state,
    )

    print(f"Rows loaded: {metrics['rows_loaded']}")
    print(f"Rows modeled: {metrics['rows_modeled']}")
    print(f"Features: {metrics['feature_columns']}")
    print(f"Train window: {metrics['train_start']} to {metrics['train_end']}")
    print(f"Test window: {metrics['test_start']} to {metrics['test_end']}")
    print(f"Train positive rate: {metrics['train_positive_rate']:.3f}")
    print(f"Test positive rate: {metrics['test_positive_rate']:.3f}")
    print(f"ROC-AUC (XGBoost): {metrics['roc_auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
