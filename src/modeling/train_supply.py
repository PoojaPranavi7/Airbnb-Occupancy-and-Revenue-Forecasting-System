# src/modeling/train_supply.py

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "features" / "model_features_with_revenue.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "supply_nights"

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def time_split(df, date_col="date", train_frac=0.7, val_frac=0.15):
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].sort_values().unique()
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df[df[date_col].isin(dates[:train_end])].copy()
    val = df[df[date_col].isin(dates[train_end:val_end])].copy()
    test = df[df[date_col].isin(dates[val_end:])].copy()
    return train, val, test

def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["neighborhood_id"] = df["neighbourhood_cleansed"].astype("category").cat.codes

    # Drop columns not used as predictors
    drop = {
        "date", "neighbourhood_cleansed",
        "occupancy_rate", "avg_price", "revenue",
        "avg_price_est", "revenue_est",
        TARGET
    }
    feature_cols = [c for c in df.columns if c not in drop]

    X = df[feature_cols]
    y = df[TARGET].astype(float)
    return X, y, feature_cols

def main():
    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[TARGET])

    train_df, val_df, test_df = time_split(df, "date", 0.7, 0.15)
    X_train, y_train, feature_cols = prepare(train_df)
    X_val, y_val, _ = prepare(val_df)
    X_test, y_test, _ = prepare(test_df)

    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2")

    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    metrics = {
        "val_mae": float(mean_absolute_error(y_val, pred_val)),
        "val_rmse": rmse(y_val, pred_val),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_rmse": rmse(y_test, pred_test),
        "n_features": int(len(feature_cols)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
    }

    joblib.dump(model, MODELS_DIR / "lgbm_supply.joblib")
    (MODELS_DIR / "supply_features.json").write_text(json.dumps(feature_cols, indent=2))
    (REPORTS_DIR / "metrics_supply.json").write_text(json.dumps(metrics, indent=2))

    print("Saved models/lgbm_supply.joblib")
    print("Saved models/supply_features.json")
    print("Saved reports/metrics_supply.json")
    print(metrics)

if __name__ == "__main__":
    main()
