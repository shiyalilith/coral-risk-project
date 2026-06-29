# Import

import os
from datetime import datetime
import csv
import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Dataset

ds = xr.open_dataset("data/raw/noaaSSTcoralData.nc")

sst = ds["analysed_sst"].squeeze()

df = sst.to_dataframe().reset_index()
df = df.rename(columns={"analysed_sst": "SST"})

df.head()


# Feature Engineering

baseline = df["SST"].mean()

df["anomaly"] = df["SST"] - baseline

df["heat_stress"] = df["anomaly"].apply(lambda x: x if x > 1 else 0)

df["rolling_stress"] = df["heat_stress"].rolling(window=84).sum()

df[["SST", "anomaly", "rolling_stress"]].head()

df["temp_change"] = df["SST"].diff()

df["temp_acceleration"] = df["temp_change"].diff()


# %%
# Prediction

df["future_anomaly"] = df["anomaly"].shift(-14)
df["risk_label"] = (df["future_anomaly"] > 1.5).astype(int)

df[["rolling_stress", "future_anomaly", "risk_label"]].tail()

# %%
# Lag Features

df["SST_lag1"] = df["SST"].shift(1)
df["SST_lag7"] = df["SST"].shift(7)
df["SST_lag14"] = df["SST"].shift(14)

df = df.dropna()

df.head()

# %%
# 1 Model training

X = df[["SST", "SST_lag1", "SST_lag7", "SST_lag14"]]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, preds))

# %%

# DISTRIBUTION SHIFT LOOP

shift_values = [0, 0.5, 1, 2]

for shift in shift_values:
    df_shifted = df.copy()
    df_shifted["SST"] += shift

    # recompute features
    baseline = df_shifted["SST"].mean()
    df_shifted["anomaly"] = df_shifted["SST"] - baseline
    df_shifted["heat_stress"] = df_shifted["anomaly"].apply(
        lambda x: x if x > 1 else 0)
    df_shifted["rolling_stress"] = df_shifted["heat_stress"].rolling(
        window=84).sum()

    df_shifted["temp_change"] = df_shifted["SST"].diff()
    df_shifted["temp_acceleration"] = df_shifted["temp_change"].diff()
    df_shifted["rolling_mean_30"] = df_shifted["SST"].rolling(30).mean()
    df_shifted["rolling_std_30"] = df_shifted["SST"].rolling(30).std()

    df_shifted["future_anomaly"] = df_shifted["anomaly"].shift(-14)
    df_shifted["risk_label"] = (df_shifted["future_anomaly"] > 1.5).astype(int)

    df_shifted["SST_lag1"] = df_shifted["SST"].shift(1)
    df_shifted["SST_lag7"] = df_shifted["SST"].shift(7)
    df_shifted["SST_lag14"] = df_shifted["SST"].shift(14)

    df_shifted = df_shifted.dropna()

    X_shifted = df_shifted[[
        "SST",
        "SST_lag1",
        "SST_lag7",
        "SST_lag14",
        "anomaly",
        "temp_change",
        "temp_acceleration",
        "rolling_mean_30",
        "rolling_std_30"
    ]]

    y_shifted = df_shifted["risk_label"]

    # retrain model on full feature set (fix feature mismatch)
    model_shift = XGBClassifier(eval_metric="logloss")
    model_shift.fit(X, y)
    preds_shifted = model_shift.predict_proba(
        X_shifted.iloc[-len(X_test):])[:, 1]
    roc_shifted = roc_auc_score(y_shifted.iloc[-len(X_test):], preds_shifted)

    print(f"Shift {shift}°C ROC-AUC:", roc_shifted)
# Visual Check

plt.figure(figsize=(12, 4))

plt.plot(df["time"].iloc[-len(preds):], y_test, label="Actual")
plt.plot(df["time"].iloc[-len(preds):], preds, label="Predicted")

plt.legend()
plt.title("Predictions vs Reality")
plt.show()

# %%
# NOTE: experimental features below are not used in final model unless retrained
# Complex

X = df[[
    "SST",
    "SST_lag1",
    "SST_lag7",
    "SST_lag14",
    "rolling_stress"
]]

# %%
X = df[[
    "SST",
    "SST_lag1",
    "SST_lag7",
    "SST_lag14",
    "anomaly",
    "rolling_stress"
]]

# %%
print("ROC-AUC:", roc_auc_score(y_test, preds))

# %%
df["rolling_mean_7"] = df["SST"].rolling(7).mean()
df["rolling_mean_14"] = df["SST"].rolling(14).mean()

# %%
X = df[[
    "SST",
    "SST_lag1",
    "SST_lag7",
    "SST_lag14",
    "anomaly",
    "rolling_stress",
    "rolling_mean_7",
    "rolling_mean_14"
]]

# %%
print("ROC-AUC:", roc_auc_score(y_test, preds))

# %%
# ===== FINAL PIPELINE (RUN THIS ONLY) =====


# Load
ds = xr.open_dataset("data/raw/noaaSSTcoralData.nc")
sst = ds["analysed_sst"].squeeze()

df = sst.to_dataframe().reset_index()
df = df.rename(columns={"analysed_sst": "SST"})

# Features
baseline = df["SST"].mean()
df["anomaly"] = df["SST"] - baseline
df["heat_stress"] = df["anomaly"].apply(lambda x: x if x > 1 else 0)
df["rolling_stress"] = df["heat_stress"].rolling(window=84).sum()

df["temp_change"] = df["SST"].diff()
df["temp_acceleration"] = df["temp_change"].diff()
df["rolling_mean_30"] = df["SST"].rolling(30).mean()
df["rolling_std_30"] = df["SST"].rolling(30).std()

# Target
df["future_anomaly"] = df["anomaly"].shift(-14)
df["risk_label"] = (df["future_anomaly"] > 1.5).astype(int)

# Lags
df["SST_lag1"] = df["SST"].shift(1)
df["SST_lag7"] = df["SST"].shift(7)
df["SST_lag14"] = df["SST"].shift(14)

df = df.dropna()

# Model
X = df[[
    "SST",
    "SST_lag1",
    "SST_lag7",
    "SST_lag14",
    "anomaly",
    "temp_change",
    "temp_acceleration",
    "rolling_mean_30",
    "rolling_std_30"
]]

y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)
preds = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, preds))

pred_labels = (preds > 0.3).astype(int)

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_labels))

# %%
model = XGBClassifier(
    eval_metric="logloss",
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1])
)

model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]

pred_labels = (preds > 0.3).astype(int)

# %%
plt.hist(preds, bins=20)
plt.title("Prediction probabilities")
plt.show()

# %%
print(confusion_matrix(y_test, pred_labels))

# %%
print("ROC-AUC:", roc_auc_score(y_test, preds))


# %%

# logging data

os.makedirs("results", exist_ok=True)
print("Current working dir:", os.getcwd())

roc = roc_auc_score(y_test, preds)

with open("results/experiments.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "2018-2024",
        "final_features",
        "+0.5",
        round(roc, 3),
        "auto-logged"
    ])
    print("Logged to CSV:", round(roc, 3))


# plot from csv


df = pd.read_csv("results/experiments.csv")

# convert shift column to numeric
df["shift"] = df["shift"].str.replace("+", "").astype(float)

# sort properly
df = df.sort_values("shift")

plt.figure(figsize=(6, 4))

plt.plot(df["shift"], df["ROC"], marker="o", linewidth=2)

plt.xlabel("Temperature Shift (°C)", fontsize=11)
plt.ylabel("ROC-AUC", fontsize=11)

plt.title("Model Robustness Under Climate-Induced Temperature Shift", fontsize=12)

# baseline random performance
plt.axhline(y=0.5, linestyle="--")

# annotate each point
for x, y in zip(df["shift"], df["ROC"]):
    plt.text(x, y, f"{y:.2f}", fontsize=9, ha='center', va='bottom')

plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.show()
