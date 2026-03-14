# Coral Risk Project

This project predicts short-horizon coral heat stress from NOAA Coral Reef Watch sea surface temperature data.

## Project Structure

- `src/data_loader.py`: load NOAA SST NetCDF data
- `src/features.py`: compute anomaly, rolling heat stress, and labels
- `src/model.py`: split, train, and evaluate the XGBoost model
- `src/train.py`: end-to-end training entry point
- `notebooks/exploration.ipynb`: lightweight exploration notebook

## Quickstart

```bash
python -m pip install -r requirements.txt
python src/healthcheck.py
python src/train.py --data data/raw/noaaSSTcoralData.nc
```

## Notes

- The pipeline uses a time-based split and avoids random shuffling.
- Model randomness is controlled through `--random-state`.
- The included dataset is a single-location daily SST series for 2024.
