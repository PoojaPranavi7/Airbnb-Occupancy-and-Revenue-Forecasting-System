# src/pipeline/build_metrics.py

import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CAL_PATH = PROCESSED_DIR / "calendar_clean.parquet"
LST_PATH = PROCESSED_DIR / "listings_clean.parquet"
OUT_PATH = PROCESSED_DIR / "daily_neighborhood_metrics.parquet"


def parse_money(series: pd.Series) -> pd.Series:
    """Convert '$123.00' -> 123.0, keep NaN."""
    if series.dtype == "O":
        return (
            series.astype(str)
            .replace({"nan": np.nan})
            .str.replace(r"[\$,]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return pd.to_numeric(series, errors="coerce")


def main():
    print("Loading cleaned parquet files...")
    cal = pd.read_parquet(CAL_PATH)
    lst = pd.read_parquet(LST_PATH)

    # Ensure types
    cal["date"] = pd.to_datetime(cal["date"])
    cal["available"] = cal["available"].astype(str).str.lower().str.strip()

    # Compute available/booked flags
    cal["is_available"] = (cal["available"] == "t").astype(int)
    cal["is_booked"] = (cal["available"] == "f").astype(int)

    # Join neighborhood + geo
    cal = cal.merge(
        lst[["id", "neighbourhood_cleansed", "latitude", "longitude", "price_clean"]],
        left_on="listing_id",
        right_on="id",
        how="left",
    )

    # Drop missing neighborhood (should be tiny after clean)
    cal = cal.dropna(subset=["neighbourhood_cleansed"])

    # -------------------------
    # Occupancy metrics (core)
    # -------------------------
    print("Building daily neighborhood occupancy metrics...")
    daily = (
        cal.groupby(["date", "neighbourhood_cleansed"], as_index=False)
        .agg(
            supply_nights=("listing_id", "count"),
            available_nights=("is_available", "sum"),
            booked_nights=("is_booked", "sum"),
        )
    )

    # booked_nights could also be supply - available; we keep both for auditing
    daily["booked_nights_check"] = daily["supply_nights"] - daily["available_nights"]
    # Use booked_nights from flag (should match check)
    daily["occupancy_rate"] = daily["booked_nights"] / daily["supply_nights"]

    # Add neighborhood centroid for geo clustering later
    neigh_geo = (
        cal.groupby("neighbourhood_cleansed", as_index=False)
        .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
    )
    daily = daily.merge(neigh_geo, on="neighbourhood_cleansed", how="left")

    # -------------------------
    # Price + Revenue (best effort)
    # -------------------------
    print("Computing avg_price + revenue (best available source)...")

    # Try calendar price first (if present)
    avg_price = None
    if "price" in cal.columns:
        cal_price = parse_money(cal["price"])
        if cal_price.notna().any():
            cal = cal.assign(price_cal=cal_price)
            # Choose mean price on available nights (or listed nights). Here: use all non-null prices.
            avg_price = (
                cal.dropna(subset=["price_cal"])
                .groupby(["date", "neighbourhood_cleansed"], as_index=False)
                .agg(avg_price=("price_cal", "mean"))
            )
            daily = daily.merge(avg_price, on=["date", "neighbourhood_cleansed"], how="left")
            daily["revenue"] = daily["booked_nights"] * daily["avg_price"]
            print("✅ Used calendar price for avg_price.")
        else:
            print("Calendar price exists but has no usable values (all/mostly NaN).")

    # Fallback: use listings price_clean (neighborhood base price)
    if "avg_price" not in daily.columns or daily["avg_price"].isna().all():
        if "price_clean" in lst.columns and lst["price_clean"].notna().any():
            neigh_price = (
                lst.dropna(subset=["price_clean"])
                .groupby("neighbourhood_cleansed", as_index=False)
                .agg(avg_price=("price_clean", "mean"))
            )
            daily = daily.merge(neigh_price, on="neighbourhood_cleansed", how="left")
            daily["revenue"] = daily["booked_nights"] * daily["avg_price"]
            print("✅ Used listings price_clean (neighborhood base price) for avg_price.")
        else:
            # No price source available right now
            daily["avg_price"] = np.nan
            daily["revenue"] = np.nan
            print("⚠️ No usable price found in calendar or listings. Revenue left as NaN for now.")

    # Sort and save
    daily = daily.sort_values(["neighbourhood_cleansed", "date"]).reset_index(drop=True)

    print(f"Saving: {OUT_PATH}")
    daily.to_parquet(OUT_PATH, index=False)

    # Summary
    print("Done ✅")
    print("Rows:", len(daily))
    print("Neighborhoods:", daily["neighbourhood_cleansed"].nunique())
    print("Date range:", daily["date"].min(), "to", daily["date"].max())

    # Quick audit check: how often booked_nights matches the check
    mismatch = (daily["booked_nights"] != daily["booked_nights_check"]).mean()
    print(f"Booked nights mismatch rate vs (supply-available): {mismatch:.6f}")


if __name__ == "__main__":
    main()
