from pathlib import Path

import pandas as pd


# Required columns from shipment_records.csv
REQUIRED_COLUMNS = {
    "origin",
    "destination",
    "route_type",
    "shipment_date",
    "quantity_tonnes",
    "distance_km",
    "freight_cost_inr",
}


def load_shipments(file_path: str | Path) -> pd.DataFrame:
    """
    Load shipment records from CSV.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Shipment file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    return df


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that all required columns are present.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def clean_shipments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning and type conversion.
    """

    df = df.copy()

    # Convert shipment_date to datetime
    df["shipment_date"] = pd.to_datetime(
        df["shipment_date"],
        errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = [
        "quantity_tonnes",
        "distance_km",
        "freight_cost_inr",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Remove rows where essential values are missing
    df = df.dropna(
        subset=[
            "origin",
            "destination",
            "route_type",
            "shipment_date",
            "quantity_tonnes",
            "distance_km",
            "freight_cost_inr",
        ]
    )

    # Remove invalid values
    df = df[
        (df["quantity_tonnes"] > 0)
        & (df["distance_km"] > 0)
        & (df["freight_cost_inr"] >= 0)
    ]

    # Remove leading/trailing spaces from text columns
    text_columns = [
        "origin",
        "destination",
        "route_type",
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    return df


def add_week_of(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add week_of column.

    week_of is the Monday of the Monday-Sunday week
    containing shipment_date.
    """

    df = df.copy()

    df["week_of"] = (
        df["shipment_date"]
        - pd.to_timedelta(
            df["shipment_date"].dt.weekday,
            unit="D"
        )
    )

    return df


def preprocess_shipments(file_path: str | Path) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.

    1. Load CSV
    2. Validate columns
    3. Clean data
    4. Convert shipment_date to datetime
    5. Create week_of
    """

    df = load_shipments(file_path)

    validate_columns(df)

    df = clean_shipments(df)

    df = add_week_of(df)

    return df