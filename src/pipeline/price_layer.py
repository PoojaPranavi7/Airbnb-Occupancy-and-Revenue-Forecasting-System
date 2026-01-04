# src/pipeline/price_layer.py

import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

DAILY_PATH = PROCESSED_DIR / "daily_neighborhood_metrics.parquet"
LISTINGS_PATH = PROCESSED_DIR / "listings_clean.parquet"

OUT_DAILY = PROCESSED_DIR / "daily_neighborhood_metrics_with_price.parquet"
OUT_FEATURES = FEATURES_DIR / "model_features_with_revenue.parquet"

def parse_money(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        return (
            s.astype(str)
             .replace({"nan": np.nan})
             .str.replace(r"[\$,]", "", regex=True)
             .replace("", np.nan)
        ).astype(float)
    return pd.to_numeric(s, errors="coerce")

def main():
    daily = pd.read_parquet(DAILY_PATH)
    lst = pd.read_parquet(LISTINGS_PATH)

    # Basic occupancy strength by neighborhood (for price adjustment)
    occ_strength = (
        daily.groupby("neighbourhood_cleansed", as_index=False)
             .agg(occ_mean=("occupancy_rate", "mean"))
    )
    occ_strength["occ_rank"] = occ_strength["occ_mean"].rank(pct=True)

    # Attempt to find a usable price column
    price_like = [c for c in lst.columns if "price" in c.lower()]
    usable_col = None
    tmp_price = None

    for c in price_like:
        parsed = parse_money(lst[c]) if c in lst.columns else None
        if parsed is not None and parsed.notna().any():
            usable_col = c
            tmp_price = parsed
            break

    if usable_col:
        lst["_price_use"] = tmp_price
        # Neighborhood base price from listings
        neigh_price = (
            lst.dropna(subset=["_price_use"])
               .groupby("neighbourhood_cleansed", as_index=False)
               .agg(base_price=(" _price_use".strip(), "mean"))
        )
    else:
        # Proxy price: room_type median + demand adjustment
        # Use room_type if present
        if "room_type" in lst.columns:
            lst_room = lst.copy()
            # Create pseudo base prices by room_type using any available signals
            # If no price at all, set reasonable defaults (NYC-style) and adjust later
            room_defaults = {
                "Entire home/apt": 220.0,
                "Private room": 110.0,
                "Shared room": 80.0,
                "Hotel room": 250.0,
            }
            lst_room["room_base"] = lst_room["room_type"].map(room_defaults).fillna(150.0)

            # Neighborhood base = mean room_base (mix of room types)
            neigh_price = (
                lst_room.groupby("neighbourhood_cleansed", as_index=False)
                        .agg(base_price=("room_base", "mean"))
            )
        else:
            # Absolute fallback (still consistent)
            neigh_price = (
                lst.groupby("neighbourhood_cleansed", as_index=False)
                   .size()
                   .rename(columns={"size":"_n"})
            )
            neigh_price["base_price"] = 150.0

    # Merge demand adjustment (higher occupancy neighborhoods slightly higher price)
    price = neigh_price.merge(occ_strength[["neighbourhood_cleansed","occ_rank"]], on="neighbourhood_cleansed", how="left")
    price["occ_rank"] = price["occ_rank"].fillna(0.5)

    # Demand multiplier: 0.85x to 1.20x
    price["demand_multiplier"] = 0.85 + 0.35 * price["occ_rank"]
    price["avg_price_est"] = price["base_price"] * price["demand_multiplier"]

    # Join into daily and compute revenue
    daily2 = daily.merge(price[["neighbourhood_cleansed","avg_price_est"]], on="neighbourhood_cleansed", how="left")
    daily2["revenue_est"] = daily2["booked_nights"] * daily2["avg_price_est"]

    daily2.to_parquet(OUT_DAILY, index=False)
    print(f"Saved: {OUT_DAILY}")

    # Also update model_features by joining revenue_est for training revenue model
    mf = pd.read_parquet(FEATURES_DIR / "model_features.parquet")
    mf2 = mf.merge(daily2[["date","neighbourhood_cleansed","avg_price_est","revenue_est"]],
                   on=["date","neighbourhood_cleansed"], how="left")

    mf2.to_parquet(OUT_FEATURES, index=False)
    print(f"Saved: {OUT_FEATURES}")

    # Quick checks
    print("avg_price_est null rate:", float(mf2["avg_price_est"].isna().mean()))
    print("revenue_est null rate:", float(mf2["revenue_est"].isna().mean()))

if __name__ == "__main__":
    main()
