# src/pipeline/features.py

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.cluster import KMeans


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

IN_PATH = PROCESSED_DIR / "daily_neighborhood_metrics.parquet"
OUT_PATH = FEATURES_DIR / "model_features.parquet"


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["dow"] = df["date"].dt.dayofweek  # 0=Mon
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    # Trend index within each neighborhood (t = 0..N)
    df = df.sort_values(["neighbourhood_cleansed", "date"])
    df["t"] = df.groupby("neighbourhood_cleansed").cumcount()

    return df


def add_lag_roll_features(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
    lags=(1, 7, 14, 28),
    rolls=(7, 14, 28),
    add_roll_std=(7, 28),
) -> pd.DataFrame:
    """
    Adds lag and rolling mean/std features for a given target column.
    """
    df = df.copy()
    g = df.groupby(group_col)[target_col]

    # Lags
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = g.shift(lag)

    # Rolling means (shift by 1 to avoid leakage from same-day target)
    for w in rolls:
        df[f"{target_col}_roll_mean_{w}"] = g.shift(1).rolling(w).mean()

    # Rolling std
    for w in add_roll_std:
        df[f"{target_col}_roll_std_{w}"] = g.shift(1).rolling(w).std()

    return df


def add_booking_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Practical booking-window proxies using occupancy dynamics.
    """
    df = df.copy()

    # availability change rate proxy: delta occ vs 7 days ago
    df["delta_occ_7"] = df["occupancy_rate"] - df["occupancy_rate_lag_7"]

    # booking momentum: short vs long moving average
    df["momentum_occ_7_28"] = df["occupancy_rate_roll_mean_7"] - df["occupancy_rate_roll_mean_28"]

    # spike flag: occ_today > roll_mean_28 + 2*roll_std_28
    df["spike_flag"] = (
        df["occupancy_rate"] > (df["occupancy_rate_roll_mean_28"] + 2 * df["occupancy_rate_roll_std_28"])
    ).astype(int)

    return df


def build_geo_clusters(df: pd.DataFrame, n_clusters: int = 8, random_state: int = 42) -> pd.DataFrame:
    """
    Cluster neighborhoods by geo + demand signature.
    Uses neighborhood-level embeddings:
      - lat/lon centroid
      - mean occupancy
      - occupancy volatility (std)
      - seasonal amplitude (max-min)
    """
    # neighborhood embeddings
    emb = (
        df.groupby("neighbourhood_cleansed", as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            occ_mean=("occupancy_rate", "mean"),
            occ_std=("occupancy_rate", "std"),
            occ_max=("occupancy_rate", "max"),
            occ_min=("occupancy_rate", "min"),
        )
    )
    emb["seasonal_amplitude"] = emb["occ_max"] - emb["occ_min"]
    emb = emb.drop(columns=["occ_max", "occ_min"])

    # Fill NaNs (std can be NaN for very short series)
    for c in ["occ_std"]:
        emb[c] = emb[c].fillna(0.0)

    # Choose features for clustering
    X = emb[["lat", "lon", "occ_mean", "occ_std", "seasonal_amplitude"]].copy()

    # Scale roughly (simple standardization without external deps)
    X = (X - X.mean()) / (X.std(ddof=0) + 1e-9)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    emb["demand_cluster_id"] = km.fit_predict(X)

    # Merge back
    df = df.merge(emb[["neighbourhood_cleansed", "demand_cluster_id"]], on="neighbourhood_cleansed", how="left")
    return df


def main():
    print(f"Loading: {IN_PATH}")
    df = pd.read_parquet(IN_PATH)

    # Basic sorting + ensure types
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    # Seasonality features
    df = add_seasonality_features(df)

    # Lags + rolling for occupancy_rate (core)
    df = add_lag_roll_features(
        df,
        group_col="neighbourhood_cleansed",
        target_col="occupancy_rate",
        lags=(1, 7, 14, 28),
        rolls=(7, 14, 28),
        add_roll_std=(7, 28),
    )

    # Price features (only if avg_price is usable)
    if "avg_price" in df.columns and df["avg_price"].notna().any():
        df = add_lag_roll_features(
            df,
            group_col="neighbourhood_cleansed",
            target_col="avg_price",
            lags=(1, 7, 14, 28),
            rolls=(7, 14, 28),
            add_roll_std=(7, 28),
        )
    else:
        print("avg_price not usable (all/mostly NaN). Skipping price lag/rolling features.")

    # Booking-window proxy features
    df = add_booking_proxy_features(df)

    # Geographic demand clusters
    df = build_geo_clusters(df, n_clusters=8)

    # Optional: Drop very early rows where lags are NaN (common approach)
    # Keep enough history to support lag_28 and roll_28 -> drop first ~28 days per neighborhood
    min_history = 28
    df["history_index"] = df.groupby("neighbourhood_cleansed").cumcount()
    df = df[df["history_index"] >= min_history].drop(columns=["history_index"])

    print(f"Saving: {OUT_PATH}")
    df.to_parquet(OUT_PATH, index=False)

    print("Done ✅")
    print("Rows:", len(df))
    print("Neighborhoods:", df["neighbourhood_cleansed"].nunique())
    print("Date range:", df["date"].min(), "to", df["date"].max())
    print("Columns:", len(df.columns))


if __name__ == "__main__":
    main()
