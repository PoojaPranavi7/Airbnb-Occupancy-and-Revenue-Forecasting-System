# src/pipeline/ingest.py

import pandas as pd
from pathlib import Path

# ------------------------
# Paths
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------
# Helpers
# ------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df

# ------------------------
# Ingest calendar
# ------------------------
def ingest_calendar():
    print("Ingesting calendar.csv...")

    df = pd.read_csv(RAW_DIR / "calendar.csv")
    df = standardize_columns(df)

    df["date"] = pd.to_datetime(df["date"])

    keep_cols = [
        "listing_id",
        "date",
        "available",
        "price",
        "adjusted_price",
        "minimum_nights",
        "maximum_nights"
    ]
    df = df[keep_cols]

    df.to_parquet(
        PROCESSED_DIR / "calendar.parquet",
        index=False
    )

    print("Saved calendar.parquet")

# ------------------------
# Ingest listings
# ------------------------
def ingest_listings():
    print("Ingesting listings.csv...")

    df = pd.read_csv(RAW_DIR / "listings.csv")
    df = standardize_columns(df)

    keep_cols = [
        "id",
        "neighbourhood_cleansed",
        "latitude",
        "longitude",
        "price",
        "room_type",
        "minimum_nights"
    ]
    df = df[keep_cols]

    df.to_parquet(
        PROCESSED_DIR / "listings.parquet",
        index=False
    )

    print("Saved listings.parquet")

# ------------------------
# Main
# ------------------------
if __name__ == "__main__":
    ingest_calendar()
    ingest_listings()
