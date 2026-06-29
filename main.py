import os
from datetime import datetime
import csv
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from xgboost import XGBClassifier

# Config
SHIFT_C = 1.0
WARMUP = 84  # longest rolling window (DHW)
FEATURE_COLS = [
    "hotspot",
    "DHW",
    "temp_change",
    "temp_acceleration",
]


def build_features(df: pd.DataFrame, mmm_map: pd.Series) -> pd.DataFrame:
    """Compute rolling/diff features. mmm_map must be fit on training data only."""
    df = df.copy()
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df["MMM"] = df["month"].map(mmm_map)
    df["hotspot"] = (df["SST"] - df["MMM"]).clip(lower=0)
    df["DHW"] = df["hotspot"].rolling(84).sum() / 7
    df["temp_change"] = df["SST"].diff()
    df["temp_acceleration"] = df["temp_change"].diff()
    df["future_DHW"] = df["DHW"].shift(-14)
    df["risk_label"] = ((df["future_DHW"] - df["DHW"]) >= 0.75).astype(int)
    return df


def make_split(
    raw_slice: pd.DataFrame,
    warmup_rows: pd.DataFrame,
    mmm_map: pd.Series,
) -> pd.DataFrame:
    """Prepend warmup rows, build features, then strip the warmup rows.

    Warmup ensures the rolling window at the split boundary is fully populated
    without contaminating evaluation rows with out-of-sample statistics.
    """
    combined = pd.concat([warmup_rows, raw_slice], ignore_index=True)
    featured = build_features(combined, mmm_map)
    return featured.iloc[len(warmup_rows):].reset_index(drop=True)


# Load
ds = xr.open_dataset("data/raw/noaaSSTcoralData.nc")
sst = ds["analysed_sst"].squeeze()
df_raw = sst.to_dataframe().reset_index()
df_raw = df_raw.rename(columns={"analysed_sst": "SST"})

# Temporal split indices (60 / 20 / 20)
n = len(df_raw)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

# Fit MMM on training rows only to avoid leakage
_train_month = df_raw.iloc[:train_end].copy()
_train_month["month"] = pd.to_datetime(_train_month["time"]).dt.month
mmm_map = _train_month.groupby("month")["SST"].mean()

# Build features per split; val/test prepend warmup rows so DHW is fully populated
_label_cols = FEATURE_COLS + ["risk_label"]

train_frame = build_features(df_raw.iloc[:train_end].copy(), mmm_map)
train_frame = train_frame.dropna(subset=_label_cols).reset_index(drop=True)

val_frame = make_split(
    df_raw.iloc[train_end:val_end],
    df_raw.iloc[max(0, train_end - WARMUP):train_end],
    mmm_map,
)
val_frame = val_frame.dropna(subset=_label_cols).reset_index(drop=True)

test_frame = make_split(
    df_raw.iloc[val_end:],
    df_raw.iloc[max(0, val_end - WARMUP):val_end],
    mmm_map,
)
test_frame = test_frame.dropna(subset=_label_cols).reset_index(drop=True)

X_train = train_frame[FEATURE_COLS]
y_train = train_frame["risk_label"]
X_val = val_frame[FEATURE_COLS]
y_val = val_frame["risk_label"]
X_test = test_frame[FEATURE_COLS]
y_test = test_frame["risk_label"]

# Model
model = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
)
model.fit(X_train, y_train)

# Select classification threshold on validation set (not test set)
val_probs = model.predict_proba(X_val)[:, 1]
_precision, _recall, _thresholds = precision_recall_curve(y_val, val_probs)
_f1 = 2 * _precision[:-1] * _recall[:-1] / \
    (_precision[:-1] + _recall[:-1] + 1e-9)
best_threshold = float(_thresholds[_f1.argmax()])

# Evaluate on held-out test set
preds = model.predict_proba(X_test)[:, 1]
pred_labels = (preds > best_threshold).astype(int)
roc = roc_auc_score(y_test, preds)

print("ROC-AUC:", roc)
print(f"Threshold (from val): {best_threshold:.3f}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_labels))

# Scenario comparison: 0.0, 0.5, 1.0 °C shifts
SCENARIOS = [0.0, 0.5, 1.0]
scenario_preds = {}
scenario_times = {}

for shift in SCENARIOS:
    df_shifted_raw = df_raw.copy()
    df_shifted_raw["SST"] += shift
    shifted_test = make_split(
        df_shifted_raw.iloc[val_end:],
        df_shifted_raw.iloc[max(0, val_end - WARMUP):val_end],
        mmm_map,
    )
    shifted_test = shifted_test.dropna(
        subset=_label_cols).reset_index(drop=True)
    X_shifted = shifted_test[FEATURE_COLS]
    y_shifted = shifted_test["risk_label"]
    common_idx = X_shifted.index.intersection(X_test.index)
    probs = model.predict_proba(X_shifted.loc[common_idx])[:, 1]
    scenario_preds[shift] = probs
    scenario_times[shift] = shifted_test.loc[common_idx, "time"].values
    print(f"ROC-AUC (+{shift}°C):",
          roc_auc_score(y_shifted.loc[common_idx], probs))

# Scenario comparison plot
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12, 4))
for shift in SCENARIOS:
    plt.plot(scenario_times[shift], scenario_preds[shift], label=f"+{shift}°C")
plt.legend()
plt.title("Predicted risk probability by warming scenario")
plt.ylabel("Predicted probability")
plt.savefig("results/scenario_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Visualise
os.makedirs("results", exist_ok=True)
plt.figure(figsize=(12, 4))
plt.plot(test_frame["time"].values, y_test.values, label="Actual")
plt.plot(test_frame["time"].values, preds, label="Predicted")
plt.legend()
plt.title("Predictions vs Reality")
plt.savefig("results/predictions_vs_reality.png", dpi=150, bbox_inches="tight")
plt.close()

plt.hist(preds, bins=20)
plt.title("Prediction probabilities")
plt.savefig("results/prediction_probabilities.png",
            dpi=150, bbox_inches="tight")
plt.close()

# Log run
os.makedirs("results", exist_ok=True)
date_range = f"{df_raw['time'].min().date()}_{df_raw['time'].max().date()}"

with open("results/experiments.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        date_range,
        ",".join(FEATURE_COLS),
        SHIFT_C,
        round(roc, 3),
        "auto-logged",
    ])
    print("Logged to CSV:", round(roc, 3))
