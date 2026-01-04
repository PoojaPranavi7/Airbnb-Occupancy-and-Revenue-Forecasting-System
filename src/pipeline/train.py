# src/pipeline/train.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "model_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "occupancy_rate"
DATE_COL = "date"
GROUP_COL = "neighbourhood_cleansed"

MODEL_PATH = MODELS_DIR / "occupancy_lgbm.joblib"
FEATURES_OUT = MODELS_DIR / "feature_list.json"
EVAL_OUT = REPORTS_DIR / "eval_occupancy.json"
IMPORTANCE_OUT = REPORTS_DIR / "feature_importance.csv"


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, eps=1e-6):
    # Avoid blow-ups when y_true near 0
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def wape(y_true, y_pred, eps=1e-6):
    denom = np.maximum(np.sum(np.abs(y_true)), eps)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def time_split_by_date(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    """
    Split by unique dates across the full dataset (time-series correct).
    """
    dates = np.sort(df[DATE_COL].unique())
    n = len(dates)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    train = df[df[DATE_COL].isin(train_dates)].copy()
    val = df[df[DATE_COL].isin(val_dates)].copy()
    test = df[df[DATE_COL].isin(test_dates)].copy()

    return train, val, test


def main():
    print(f"Loading features: {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)

    # Ensure datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    # Encode neighbourhood as numeric id (global model needs this)
    df[GROUP_COL] = df[GROUP_COL].astype("category")
    df["neighborhood_id"] = df[GROUP_COL].cat.codes.astype(np.int32)

    # Ensure cluster id numeric
    if "demand_cluster_id" in df.columns:
        df["demand_cluster_id"] = pd.to_numeric(df["demand_cluster_id"], errors="coerce").fillna(-1).astype(int)

    # Define feature columns
    drop_cols = {
        TARGET,
        DATE_COL,
        GROUP_COL,
        # optional columns we don't want as features:
        "booked_nights_check",  # if present from metrics stage
        "avg_price", "revenue",  # currently NaN; exclude from occupancy model
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Remove any non-numeric columns (safety)
    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        print("Dropping non-numeric feature columns:", non_numeric)
        feature_cols = [c for c in feature_cols if c not in non_numeric]

    # Drop rows with missing target or critical features
    df = df.dropna(subset=[TARGET])
    df = df.sort_values([DATE_COL, "neighborhood_id"]).reset_index(drop=True)

    # Time split
    train_df, val_df, test_df = time_split_by_date(df, train_frac=0.70, val_frac=0.15)

    print("Split sizes:")
    print("  Train:", len(train_df))
    print("  Val  :", len(val_df))
    print("  Test :", len(test_df))
    print("Date ranges:")
    print("  Train:", train_df[DATE_COL].min(), "to", train_df[DATE_COL].max())
    print("  Val  :", val_df[DATE_COL].min(), "to", val_df[DATE_COL].max())
    print("  Test :", test_df[DATE_COL].min(), "to", test_df[DATE_COL].max())

    X_train, y_train = train_df[feature_cols], train_df[TARGET]
    X_val, y_val = val_df[feature_cols], val_df[TARGET]
    X_test, y_test = test_df[feature_cols], test_df[TARGET]

    # LightGBM model (strong default for tabular forecasting)
    model = LGBMRegressor(
        n_estimators=5000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    print("Training LightGBM...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        callbacks=[],
    )

    # Predict
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    # Clip predictions to valid occupancy range [0, 1]
    val_pred = np.clip(val_pred, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    # Metrics
    results = {
        "target": TARGET,
        "features_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "val": {
            "MAE": mae(y_val.values, val_pred),
            "RMSE": rmse(y_val.values, val_pred),
            "MAPE_%": mape(y_val.values, val_pred),
            "WAPE_%": wape(y_val.values, val_pred),
        },
        "test": {
            "MAE": mae(y_test.values, test_pred),
            "RMSE": rmse(y_test.values, test_pred),
            "MAPE_%": mape(y_test.values, test_pred),
            "WAPE_%": wape(y_test.values, test_pred),
        },
    }

    print("Validation metrics:", results["val"])
    print("Test metrics      :", results["test"])

    # Save model + features list
    dump(model, MODEL_PATH)
    with open(FEATURES_OUT, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    with open(EVAL_OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    fi.to_csv(IMPORTANCE_OUT, index=False)

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved feature list: {FEATURES_OUT}")
    print(f"Saved eval report: {EVAL_OUT}")
    print(f"Saved feature importance: {IMPORTANCE_OUT}")
    print("Done ✅")


if __name__ == "__main__":
    main()
