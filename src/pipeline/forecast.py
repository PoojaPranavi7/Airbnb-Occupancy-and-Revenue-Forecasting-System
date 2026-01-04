# src/pipeline/forecast.py

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FORECAST_DIR = PROJECT_ROOT / "data" / "forecasts"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_PATH = PROCESSED_DIR / "daily_neighborhood_metrics_with_price.parquet"

OCC_MODEL_PATH = MODELS_DIR / "lgbm_occupancy.joblib"
REV_MODEL_PATH = MODELS_DIR / "lgbm_revenue.joblib"

OCC_FEATURES_PATH = MODELS_DIR / "occupancy_features.json"
REV_FEATURES_PATH = MODELS_DIR / "revenue_features.json"

OUTPUT_PATH = FORECAST_DIR / "neighborhood_forecast_90d.parquet"

HORIZON_DAYS = 90


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # t: time index per neighborhood
    df = df.sort_values(["neighbourhood_cleansed", "date"])
    df["t"] = df.groupby("neighbourhood_cleansed").cumcount()
    return df


def add_lag_roll_features(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(group_col)[target_col]

    # lags used in training
    for lag in (1, 7, 14, 28):
        df[f"{target_col}_lag_{lag}"] = g.shift(lag)

    # rolling means (shift 1 day to prevent leakage)
    for w in (7, 14, 28):
        df[f"{target_col}_roll_mean_{w}"] = g.shift(1).rolling(w).mean()

    # rolling std
    for w in (7, 28):
        df[f"{target_col}_roll_std_{w}"] = g.shift(1).rolling(w).std()

    return df


def add_booking_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_occ_7"] = df["occupancy_rate"] - df["occupancy_rate_lag_7"]
    df["momentum_occ_7_28"] = df["occupancy_rate_roll_mean_7"] - df["occupancy_rate_roll_mean_28"]
    df["spike_flag"] = (
        df["occupancy_rate"] > (df["occupancy_rate_roll_mean_28"] + 2 * df["occupancy_rate_roll_std_28"])
    ).astype(int)
    return df


def build_neighborhood_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neighborhood_id"] = df["neighbourhood_cleansed"].astype("category").cat.codes
    return df


def main():
    print("Loading history...")
    hist = pd.read_parquet(HISTORY_PATH)
    hist["date"] = pd.to_datetime(hist["date"])

    # Ensure needed columns exist
    required_cols = ["date", "neighbourhood_cleansed", "occupancy_rate", "booked_nights", "supply_nights", "lat", "lon", "avg_price_est", "revenue_est"]
    missing = [c for c in required_cols if c not in hist.columns]
    if missing:
        raise ValueError(f"Missing required columns in history: {missing}")

    # Load models + feature lists
    occ_model = joblib.load(OCC_MODEL_PATH)
    rev_model = joblib.load(REV_MODEL_PATH)
    occ_features = json.loads(Path(OCC_FEATURES_PATH).read_text())
    rev_features = json.loads(Path(REV_FEATURES_PATH).read_text())

    # Demand cluster id exists in your features dataset, but not guaranteed in history.
    # We'll compute a simple stable cluster assignment using precomputed centroids if present.
    # If you don't have demand_cluster_id in history, we create a placeholder 0.
    if "demand_cluster_id" not in hist.columns:
        hist["demand_cluster_id"] = 0

    # We will forecast per neighborhood iteratively
    last_date = hist["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    # Keep only the columns we need and ensure sorted
    hist = hist.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    all_forecasts = []

    neighborhoods = hist["neighbourhood_cleansed"].unique()
    print("Neighborhoods:", len(neighborhoods))
    print("History last date:", last_date.date())
    print("Forecast horizon:", HORIZON_DAYS, "days")

    # For iterative updates, we maintain a working dataframe per neighborhood with past + predicted occupancy
    for nb in neighborhoods:
        nb_hist = hist[hist["neighbourhood_cleansed"] == nb].copy()
        nb_hist = nb_hist.sort_values("date").reset_index(drop=True)

        # We'll keep a series of occupancy_rate values (past + predicted)
        # Start with existing
        work = nb_hist.copy()

        # Iterate forward day by day
        for d in future_dates:
            # Create one new row for date d (unknown target values)
            new_row = {
                "date": d,
                "neighbourhood_cleansed": nb,
                "supply_nights": float(work["supply_nights"].iloc[-1]),  # carry forward last known supply
                "booked_nights": np.nan,
                "occupancy_rate": np.nan,
                "lat": float(work["lat"].iloc[-1]),
                "lon": float(work["lon"].iloc[-1]),
                "avg_price_est": float(work["avg_price_est"].iloc[-1]),
                "revenue_est": np.nan,
                "demand_cluster_id": int(work["demand_cluster_id"].iloc[-1]),
            }

            work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

            # Recompute engineered features on the full work dataframe
            work = add_time_features(work)
            work = add_lag_roll_features(work, "neighbourhood_cleansed", "occupancy_rate")
            work = add_booking_proxy_features(work)
            work = build_neighborhood_ids(work)

            # Select current row features for prediction (last row)
            cur = work.iloc[[-1]].copy()

            # Fill NaNs in lag/roll features (should be minimal because we have history)
            # LightGBM can handle NaNs, but we can keep as-is.
            X_occ = cur.reindex(columns=occ_features)

            # Predict occupancy
            occ_pred = float(occ_model.predict(X_occ)[0])
            # Clip to [0,1]
            occ_pred = float(np.clip(occ_pred, 0.0, 1.0))

            # Update occupancy + booked nights for the new row
            work.loc[work.index[-1], "occupancy_rate"] = occ_pred
            supply = float(work.loc[work.index[-1], "supply_nights"])
            booked = occ_pred * supply
            work.loc[work.index[-1], "booked_nights"] = booked

            # Prepare revenue features and predict revenue_est
            # We keep avg_price_est in features; revenue model learned from engineered revenue_est.
            X_rev = work.iloc[[-1]].reindex(columns=rev_features)
            rev_pred = float(rev_model.predict(X_rev)[0])

            # Revenue can't be negative
            rev_pred = float(max(0.0, rev_pred))
            work.loc[work.index[-1], "revenue_est"] = rev_pred

        # Collect only the forecast rows (future_dates)
        nb_forecast = work[work["date"].isin(future_dates)].copy()
        nb_forecast["forecast_horizon_days"] = HORIZON_DAYS
        all_forecasts.append(nb_forecast)

    forecast_df = pd.concat(all_forecasts, ignore_index=True)

    # Keep only important output columns for BI/Streamlit
    keep = [
        "date",
        "neighbourhood_cleansed",
        "demand_cluster_id",
        "lat",
        "lon",
        "supply_nights",
        "occupancy_rate",
        "booked_nights",
        "avg_price_est",
        "revenue_est",
        "forecast_horizon_days",
    ]
    forecast_df = forecast_df[keep].sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    print(f"Saving forecast: {OUTPUT_PATH}")
    forecast_df.to_parquet(OUTPUT_PATH, index=False)

    print("Done ✅")
    print("Forecast rows:", len(forecast_df))
    print("Forecast date range:", forecast_df["date"].min(), "to", forecast_df["date"].max())


if __name__ == "__main__":
    main()
