# Coral Risk Project

## Overview

This project investigates how machine learning models for coral heat stress prediction behave under climate-induced distribution shift.

Using NOAA Coral Reef Watch sea surface temperature (SST) data, we train a model to predict short-horizon thermal stress risk and evaluate how its performance degrades as ocean temperatures increase.

---

## Key Question

**How robust are SST-based coral stress prediction models under changing climate conditions?**

Rather than focusing only on predictive accuracy, this project analyzes **generalization and failure under distribution shift**.

---

## Main Results

- In-distribution performance:
  - ROC-AUC ≈ **0.91**

- Under temperature shifts:
  - +0.5°C → ~0.89
  - +1.0°C → ~0.84
  - +2.0°C → ~0.60

### Insight

Model performance degrades **non-linearly** under increasing temperature shifts.

This suggests that the model relies on **absolute SST thresholds**, rather than learning invariant stress dynamics. Under large climate shifts, these thresholds become invalid.

---

## Method

### Data

- NOAA Coral Reef Watch SST (NetCDF)
- Multi-year daily temperature data

### Features

- SST (raw)
- Lagged SST (1, 7, 14 days)
- Derived anomaly and temporal features (final pipeline)

### Target

- Binary classification:
  - Future anomaly > 1.5 (14-day horizon)

### Model

- XGBoost classifier
- Time-based train/test split (no shuffling)

---

## Experiments

### 1. Baseline Prediction

Train on historical SST → predict future anomaly.

### 2. Distribution Shift Testing

Simulate warming scenarios by shifting SST:

- +0.5°C
- +1.0°C
- +2.0°C

Evaluate model performance **without retraining**.

---

## Key Findings

- Strong performance under current conditions
- Robust to mild warming
- Significant breakdown under extreme shift (+2°C)

### Interpretation

Models trained on historical environmental data may fail under future climate conditions due to **distribution shift**.

---

## Project Structure

- `src/data_loader.py` – load NOAA SST data
- `src/features.py` – feature engineering
- `src/model.py` – training and evaluation
- `src/train.py` – pipeline entry point
- `docs/research_log.md` – full research progression
- `results/` – experiment tracking and outputs

---

## Quickstart

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
make check
make run
```

---

## Why This Matters

Climate systems are **non-stationary**. Models trained on past data may not generalize to future conditions.

This project demonstrates a concrete example of:

- distribution shift
- model fragility
- limits of static ML systems in changing environments

---

## Next Steps

- Cross-reef generalization (multiple locations)
- Temporal generalization (train past → predict future years)
- Investigate invariant feature representations
- Explore model recalibration under shift
