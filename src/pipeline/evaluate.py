# src/pipeline/evaluate.py

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = PROJECT_ROOT / "data" / "features" / "model_features_with_revenue.parquet"

MODELS_DIR = PROJECT_ROOT / "models"
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

OCC_MODEL_PATH = MODELS_DIR / "lgbm_occupancy.joblib"
REV_MODEL_PATH = MODELS_DIR / "lgbm_revenue.joblib"
SUP_MODEL_PATH = MODELS_DIR / "lgbm_supply.joblib"

OCC_FEATURES_PATH = MODELS_DIR / "occupancy_features.json"
REV_FEATURES_PATH = MODELS_DIR / "revenue_features.json"
SUP_FEATURES_PATH = MODELS_DIR / "supply_features.json"

OUT_JSON = FORECAST_DIR / "metrics.json"
OUT_TABLE = FORECAST_DIR / "metrics_table.parquet"


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(eps, (np.abs(y_true) + np.abs(y_pred)) / 2.0)
    return float(np.mean(np.abs(y_pred - y_true) / denom))


def wape(y_true, y_pred, eps=1e-9) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(eps, np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def time_split(df: pd.DataFrame, date_col="date", train_frac=0.7, val_frac=0.15):
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].sort_values().unique()
    n = len(dates)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    return train_dates, val_dates, test_dates


def prepare_Xy(df: pd.DataFrame, feature_cols: list, target_col: str):
    df = df.copy()
    df["neighborhood_id"] = df["neighbourhood_cleansed"].astype("category").cat.codes
    X = df.reindex(columns=feature_cols)
    y = df[target_col].astype(float).values
    return X, y


def main():
    # Load dataset (model-ready features)
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    # Required columns
    required = ["occupancy_rate", "supply_nights", "revenue_est", "occupancy_rate_lag_7"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in features dataset: {missing}")

    # Load models + feature lists
    occ_model = joblib.load(OCC_MODEL_PATH)
    rev_model = joblib.load(REV_MODEL_PATH)
    sup_model = joblib.load(SUP_MODEL_PATH)

    occ_features = json.loads(Path(OCC_FEATURES_PATH).read_text())
    rev_features = json.loads(Path(REV_FEATURES_PATH).read_text())
    sup_features = json.loads(Path(SUP_FEATURES_PATH).read_text())

    # Time split (same as training policy)
    _, _, test_dates = time_split(df, "date", 0.7, 0.15)
    test_df = df[df["date"].isin(test_dates)].copy()
    test_df = test_df.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    # -------------------------
    # Model predictions (one-step evaluation)
    # -------------------------
    X_occ, y_occ = prepare_Xy(test_df, occ_features, "occupancy_rate")
    occ_pred = np.clip(occ_model.predict(X_occ), 0.0, 1.0)

    X_sup, y_sup = prepare_Xy(test_df, sup_features, "supply_nights")
    sup_pred = np.maximum(0.0, sup_model.predict(X_sup))

    X_rev, y_rev = prepare_Xy(test_df, rev_features, "revenue_est")
    rev_pred = np.maximum(0.0, rev_model.predict(X_rev))

    # -------------------------
    # Baselines (seasonal naive: t-7 days)
    # -------------------------
    # Occupancy baseline: use precomputed lag_7 (already in features)
    occ_base = test_df["occupancy_rate_lag_7"].astype(float).values

    # Revenue baseline: TRUE seasonal naive on revenue_est shifted by 7 *within each neighborhood*
    rev_base = (
        test_df.groupby("neighbourhood_cleansed")["revenue_est"]
        .shift(7)
        .astype(float)
        .values
    )

    # For revenue baseline, first 7 rows per neighborhood will be NaN -> remove those from BOTH arrays
    rev_mask = ~np.isnan(rev_base)
    y_rev_base = y_rev[rev_mask]
    rev_pred_base = rev_pred[rev_mask]
    rev_base_clean = rev_base[rev_mask]

    # -------------------------
    # Metrics
    # -------------------------
    results = {
        "occupancy_model": {
            "mae": float(mean_absolute_error(y_occ, occ_pred)),
            "rmse": rmse(y_occ, occ_pred),
            "smape": smape(y_occ, occ_pred),
        },
        "occupancy_baseline_seasonal_naive": {
            "mae": float(mean_absolute_error(y_occ, occ_base)),
            "rmse": rmse(y_occ, occ_base),
            "smape": smape(y_occ, occ_base),
        },
        "supply_model": {
            "mae": float(mean_absolute_error(y_sup, sup_pred)),
            "rmse": rmse(y_sup, sup_pred),
        },
        "revenue_model": {
            "mae": float(mean_absolute_error(y_rev, rev_pred)),
            "rmse": rmse(y_rev, rev_pred),
            "wape": wape(y_rev, rev_pred),
        },
        "revenue_baseline_seasonal_naive": {
            "mae": float(mean_absolute_error(y_rev_base, rev_base_clean)),
            "rmse": rmse(y_rev_base, rev_base_clean),
            "wape": wape(y_rev_base, rev_base_clean),
            "rows_used": int(len(y_rev_base)),
            "rows_dropped_due_to_lag7_nan": int(len(y_rev) - len(y_rev_base)),
        },
        "test_window": {
            "rows": int(len(test_df)),
            "date_min": str(test_df["date"].min()),
            "date_max": str(test_df["date"].max()),
            "neighborhoods": int(test_df["neighbourhood_cleansed"].nunique()),
        },
    }

    # Save outputs
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUT_JSON}")

    metrics_table = pd.DataFrame([
        {"target": "occupancy", "method": "model", **results["occupancy_model"]},
        {"target": "occupancy", "method": "seasonal_naive", **results["occupancy_baseline_seasonal_naive"]},
        {"target": "supply", "method": "model", **results["supply_model"]},
        {"target": "revenue", "method": "model", **results["revenue_model"]},
        {"target": "revenue", "method": "seasonal_naive", **results["revenue_baseline_seasonal_naive"]},
    ])
    metrics_table.to_parquet(OUT_TABLE, index=False)
    print(f"Saved: {OUT_TABLE}")

    print("Evaluation results ✅")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
