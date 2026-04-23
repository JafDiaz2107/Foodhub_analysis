"""Data loading and validation for the FoodHub order dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_COLUMNS = [
    "order_id",
    "customer_id",
    "restaurant_name",
    "cuisine_type",
    "cost_of_the_order",
    "day_of_the_week",
    "rating",
    "food_preparation_time",
    "delivery_time",
]


def load_foodhub(filepath: str | Path) -> pd.DataFrame:
    """Load the FoodHub order dataset from a CSV file.

    Converts the 'rating' column from mixed string/int to float,
    replacing "Not given" with NaN.

    Args:
        filepath: Path to the foodhub_order.csv file.

    Returns:
        DataFrame with cleaned types.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)
    validate_schema(df)

    df["rating"] = df["rating"].replace("Not given", np.nan).astype(float)

    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that the DataFrame has the expected FoodHub columns.

    Args:
        df: DataFrame to validate.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return True
