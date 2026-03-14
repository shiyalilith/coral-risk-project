"""Model training and evaluation helpers."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

try:
    from .features import feature_columns
except ImportError:
    from features import feature_columns


def find_time_split_index(
    labels: pd.Series,
    min_train_size: int = 60,
    min_test_size: int = 30,
    min_class_count: int = 5,
) -> int:
    """Find the latest chronological split with both classes on each side."""
    valid_splits: list[int] = []

    for split_index in range(min_train_size, len(labels) - min_test_size + 1):
        train = labels.iloc[:split_index]
        test = labels.iloc[split_index:]

        train_pos = int(train.sum())
        test_pos = int(test.sum())
        train_neg = len(train) - train_pos
        test_neg = len(test) - test_pos

        if min(train_pos, test_pos, train_neg, test_neg) >= min_class_count:
            valid_splits.append(split_index)

    if not valid_splits:
        raise ValueError("Unable to find a valid time-based split with both classes in train and test")

    return valid_splits[-1]


def split_train_test(model_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the model frame chronologically."""
    split_index = find_time_split_index(model_frame["future_risk"])
    train_frame = model_frame.iloc[:split_index].copy()
    test_frame = model_frame.iloc[split_index:].copy()
    return train_frame, test_frame


def train_xgboost(train_frame: pd.DataFrame, random_state: int) -> XGBClassifier:
    """Train the XGBoost classifier on the engineered features."""
    positive_count = int(train_frame["future_risk"].sum())
    negative_count = len(train_frame) - positive_count
    scale_pos_weight = negative_count / positive_count if positive_count else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(train_frame[feature_columns()], train_frame["future_risk"])
    return model


def evaluate_model(
    model: XGBClassifier,
    test_frame: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate the model on the held-out chronological test set."""
    probabilities = model.predict_proba(test_frame[feature_columns()])[:, 1]
    roc_auc = roc_auc_score(test_frame["future_risk"], probabilities)
    return {
        "roc_auc": roc_auc,
        "test_positive_rate": test_frame["future_risk"].mean(),
        "test_start": test_frame["time"].min(),
        "test_end": test_frame["time"].max(),
    }
