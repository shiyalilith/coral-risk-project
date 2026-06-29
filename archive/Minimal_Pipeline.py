# Coral Risk Minimal Pipeline

import xarray as xr
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# ===== LOAD DATA =====
ds = xr.open_dataset("data/raw/noaaSSTcoralData.nc")
sst = ds["analysed_sst"].squeeze()

df = sst.to_dataframe().reset_index()
df = df.rename(columns={"analysed_sst": "SST"})

# ===== FEATURE ENGINEERING =====
baseline = df["SST"].mean()
df["anomaly"] = df["SST"] - baseline

df["heat_stress"] = df["anomaly"].apply(lambda x: x if x > 1 else 0)
df["rolling_stress"] = df["heat_stress"].rolling(window=84).sum()

# ===== LABEL (FUTURE RISK) =====
df["future_stress"] = df["rolling_stress"].shift(-14)
df["risk_label"] = (df["future_stress"] > 4).astype(int)

# ===== LAG FEATURES =====
df["SST_lag1"] = df["SST"].shift(1)
df["SST_lag7"] = df["SST"].shift(7)
df["SST_lag14"] = df["SST"].shift(14)

df = df.dropna()

# ===== MODEL =====
X = df[["SST", "SST_lag1", "SST_lag7", "SST_lag14"]]
y = df["risk_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]

print("ROC-AUC:", roc_auc_score(y_test, preds))

# ===== QUICK CHECK =====
print(df.head())
