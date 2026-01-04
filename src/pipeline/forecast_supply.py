# src/pipeline/forecast_supply.py

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
SUP_MODEL_PATH = MODELS_DIR / "lgbm_supply.joblib"

OCC_FEATURES_PATH = MODELS_DIR / "occupancy_features.json"
REV_FEATURES_PATH = MODELS_DIR / "revenue_features.json"
SUP_FEATURES_PATH = MODELS_DIR / "supply_features.json"

OUTPUT_PATH = FORECAST_DIR / "neighborhood_forecast_90d_supply.parquet"

HORIZON_DAYS = 90


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df = df.sort_values(["neighbourhood_cleansed", "date"])
    df["t"] = df.groupby("neighbourhood_cleansed").cumcount()
    return df


def add_lag_roll(df: pd.DataFrame, group_col: str, col: str) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(group_col)[col]

    for lag in (1, 7, 14, 28):
        df[f"{col}_lag_{lag}"] = g.shift(lag)

    for w in (7, 14, 28):
        df[f"{col}_roll_mean_{w}"] = g.shift(1).rolling(w).mean()

    for w in (7, 28):
        df[f"{col}_roll_std_{w}"] = g.shift(1).rolling(w).std()

    return df


def add_booking_proxy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_occ_7"] = df["occupancy_rate"] - df["occupancy_rate_lag_7"]
    df["momentum_occ_7_28"] = df["occupancy_rate_roll_mean_7"] - df["occupancy_rate_roll_mean_28"]
    df["spike_flag"] = (
        df["occupancy_rate"] > (df["occupancy_rate_roll_mean_28"] + 2 * df["occupancy_rate_roll_std_28"])
    ).astype(int)
    return df


def add_neighborhood_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["neighborhood_id"] = df["neighbourhood_cleansed"].astype("category").cat.codes
    return df


def main():
    print("Loading history...")
    hist = pd.read_parquet(HISTORY_PATH)
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    if "demand_cluster_id" not in hist.columns:
        hist["demand_cluster_id"] = 0

    # Load models and feature lists
    occ_model = joblib.load(OCC_MODEL_PATH)
    rev_model = joblib.load(REV_MODEL_PATH)
    sup_model = joblib.load(SUP_MODEL_PATH)

    occ_features = json.loads(Path(OCC_FEATURES_PATH).read_text())
    rev_features = json.loads(Path(REV_FEATURES_PATH).read_text())
    sup_features = json.loads(Path(SUP_FEATURES_PATH).read_text())

    last_date = hist["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=HORIZON_DAYS, freq="D")

    neighborhoods = hist["neighbourhood_cleansed"].unique()
    print("Neighborhoods:", len(neighborhoods))
    print("History last date:", last_date.date())
    print("Forecast horizon:", HORIZON_DAYS, "days")

    all_forecasts = []

    for nb in neighborhoods:
        work = hist[hist["neighbourhood_cleansed"] == nb].copy().reset_index(drop=True)

        for d in future_dates:
            # Add blank future row
            new_row = {
                "date": d,
                "neighbourhood_cleansed": nb,
                # placeholders; we will predict supply + occupancy
                "supply_nights": np.nan,
                "booked_nights": np.nan,
                "occupancy_rate": np.nan,
                "lat": float(work["lat"].iloc[-1]),
                "lon": float(work["lon"].iloc[-1]),
                "avg_price_est": float(work["avg_price_est"].iloc[-1]),
                "revenue_est": np.nan,
                "demand_cluster_id": int(work["demand_cluster_id"].iloc[-1]),
            }

            work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)

            # Recompute features (includes past + predicted rows)
            work = add_time_features(work)

            # supply features (lags/rolls on supply)
            work = add_lag_roll(work, "neighbourhood_cleansed", "supply_nights")

            # occupancy features (lags/rolls on occupancy)
            work = add_lag_roll(work, "neighbourhood_cleansed", "occupancy_rate")
            work = add_booking_proxy(work)

            work = add_neighborhood_id(work)

            cur = work.iloc[[-1]].copy()

            # ---- Predict SUPPLY ----
            X_sup = cur.reindex(columns=sup_features)
            sup_pred = float(sup_model.predict(X_sup)[0])
            sup_pred = max(0.0, sup_pred)
            # supply nights should be an integer count
            sup_pred = float(round(sup_pred))

            work.loc[work.index[-1], "supply_nights"] = sup_pred

            # ---- Predict OCCUPANCY ----
            X_occ = work.iloc[[-1]].reindex(columns=occ_features)
            occ_pred = float(occ_model.predict(X_occ)[0])
            occ_pred = float(np.clip(occ_pred, 0.0, 1.0))

            work.loc[work.index[-1], "occupancy_rate"] = occ_pred

            # booked nights derived from occ × supply
            booked = occ_pred * sup_pred
            work.loc[work.index[-1], "booked_nights"] = booked

            # ---- Predict REVENUE ----
            X_rev = work.iloc[[-1]].reindex(columns=rev_features)
            rev_pred = float(rev_model.predict(X_rev)[0])
            rev_pred = max(0.0, rev_pred)
            work.loc[work.index[-1], "revenue_est"] = rev_pred

        nb_fc = work[work["date"].isin(future_dates)].copy()
        nb_fc["forecast_horizon_days"] = HORIZON_DAYS
        all_forecasts.append(nb_fc)

    fc = pd.concat(all_forecasts, ignore_index=True)

    keep = [
        "date",
        "neighbourhood_cleansed",
        "demand_cluster_id",
        "lat", "lon",
        "supply_nights",
        "occupancy_rate",
        "booked_nights",
        "avg_price_est",
        "revenue_est",
        "forecast_horizon_days",
    ]
    fc = fc[keep].sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    print(f"Saving forecast: {OUTPUT_PATH}")
    fc.to_parquet(OUTPUT_PATH, index=False)

    print("Done ✅")
    print("Forecast rows:", len(fc))
    print("Forecast date range:", fc["date"].min(), "to", fc["date"].max())


if __name__ == "__main__":
    main()
