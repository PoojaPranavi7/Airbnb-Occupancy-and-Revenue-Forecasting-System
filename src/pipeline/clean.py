# src/pipeline/clean.py

import pandas as pd
import numpy as np
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CAL_IN = PROCESSED_DIR / "calendar.parquet"
LST_IN = PROCESSED_DIR / "listings.parquet"

CAL_OUT = PROCESSED_DIR / "calendar_clean.parquet"
LST_OUT = PROCESSED_DIR / "listings_clean.parquet"


def parse_money(series: pd.Series) -> pd.Series:
    """
    Convert strings like '$120.00' to float. Keeps NaN if missing.
    """
    if series.dtype == "O":
        return (
            series.astype(str)
            .replace({"nan": np.nan})
            .str.replace(r"[\$,]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
    return series.astype(float)


def winsorize_series(s: pd.Series, lower_q=0.01, upper_q=0.99) -> pd.Series:
    """
    Cap extreme values to reduce outlier impact.
    """
    s = s.copy()
    lo = s.quantile(lower_q)
    hi = s.quantile(upper_q)
    return s.clip(lower=lo, upper=hi)


def clean_calendar() -> pd.DataFrame:
    print("Loading calendar.parquet...")
    cal = pd.read_parquet(CAL_IN)

    # Dates
    cal["date"] = pd.to_datetime(cal["date"], errors="coerce")
    cal = cal.dropna(subset=["date"])

    # Normalize available: keep only 't'/'f'
    cal["available"] = cal["available"].astype(str).str.lower().str.strip()
    cal = cal[cal["available"].isin(["t", "f"])]

    # Nights columns sanity
    for col in ["minimum_nights", "maximum_nights"]:
        if col in cal.columns:
            cal[col] = pd.to_numeric(cal[col], errors="coerce")

    # Drop impossible nights (optional but safe)
    if "minimum_nights" in cal.columns:
        cal = cal[(cal["minimum_nights"].isna()) | (cal["minimum_nights"] >= 1)]
    if "maximum_nights" in cal.columns:
        cal = cal[(cal["maximum_nights"].isna()) | (cal["maximum_nights"] >= 1)]

    # Parse prices (calendar price can be mostly NaN; that's fine)
    if "price" in cal.columns:
        cal["price"] = parse_money(cal["price"])
    if "adjusted_price" in cal.columns:
        cal["adjusted_price"] = parse_money(cal["adjusted_price"])

    # Ensure consistent date range (optional: keep only overlapping range)
    # For forecasting, it's okay to keep all. But we often limit to a stable window.
    # Here we keep everything, but you can uncomment to constrain:
    # start, end = cal["date"].min(), cal["date"].max()
    # cal = cal[(cal["date"] >= start) & (cal["date"] <= end)]

    # Remove duplicate rows if any
    cal = cal.drop_duplicates(subset=["listing_id", "date"])

    print(f"Calendar cleaned: {len(cal):,} rows")
    return cal


def clean_listings() -> pd.DataFrame:
    print("Loading listings.parquet...")
    lst = pd.read_parquet(LST_IN)

    # Drop rows missing neighborhood
    lst["neighbourhood_cleansed"] = lst["neighbourhood_cleansed"].astype(str).str.strip()
    lst = lst[lst["neighbourhood_cleansed"].notna()]
    lst = lst[lst["neighbourhood_cleansed"].str.lower() != "nan"]
    lst = lst[lst["neighbourhood_cleansed"] != ""]

    # Parse listing-level price (this will be our revenue proxy)
    if "price" in lst.columns:
        lst["price_clean"] = parse_money(lst["price"])
    else:
        lst["price_clean"] = np.nan

    # Cap extreme listing prices (winsorize)
    if lst["price_clean"].notna().any():
        lst["price_clean"] = winsorize_series(lst["price_clean"], 0.01, 0.99)

    # Lat/lon sanity filters (NYC-ish ranges are roughly lat 40-41, lon -75 to -73,
    # but we keep broader to avoid dropping valid edges)
    lst["latitude"] = pd.to_numeric(lst["latitude"], errors="coerce")
    lst["longitude"] = pd.to_numeric(lst["longitude"], errors="coerce")
    lst = lst.dropna(subset=["latitude", "longitude"])
    lst = lst[(lst["latitude"].between(-90, 90)) & (lst["longitude"].between(-180, 180))]

    # Remove duplicates by listing id
    lst = lst.drop_duplicates(subset=["id"])

    print(f"Listings cleaned: {len(lst):,} rows")
    return lst


def main():
    cal = clean_calendar()
    lst = clean_listings()

    print("Saving cleaned parquet files...")
    cal.to_parquet(CAL_OUT, index=False)
    lst.to_parquet(LST_OUT, index=False)

    print(f"Saved: {CAL_OUT}")
    print(f"Saved: {LST_OUT}")


if __name__ == "__main__":
    main()
