"""Feature engineering for FoodHub order analysis.

Pipeline order: clean → engineer → encode
No scaling step needed (tree-based models handle raw features;
statistical tests use original distributions).
"""

import numpy as np
import pandas as pd

NUMERIC_COLS = ["cost_of_the_order", "food_preparation_time", "delivery_time"]
CATEGORICAL_COLS = ["cuisine_type", "day_of_the_week"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the loaded FoodHub dataset.

    Adds a has_rating flag and validates value ranges.

    Args:
        df: DataFrame from load_foodhub() (rating already float/NaN).

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    df["has_rating"] = df["rating"].notna().astype(int)

    if (df["cost_of_the_order"] < 0).any():
        raise ValueError("Negative order costs found")
    if (df["food_preparation_time"] < 0).any():
        raise ValueError("Negative preparation times found")
    if (df["delivery_time"] < 0).any():
        raise ValueError("Negative delivery times found")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features for analysis and modeling.

    Args:
        df: Cleaned DataFrame (after clean_data).

    Returns:
        DataFrame with additional engineered features.
    """
    df = df.copy()

    df["total_time"] = df["food_preparation_time"] + df["delivery_time"]
    df["cost_per_minute"] = np.where(
        df["total_time"] > 0,
        df["cost_of_the_order"] / df["total_time"],
        0.0,
    )
    df["prep_delivery_ratio"] = np.where(
        df["delivery_time"] > 0,
        df["food_preparation_time"] / df["delivery_time"],
        0.0,
    )
    df["is_weekend"] = (df["day_of_the_week"] == "Weekend").astype(int)

    order_counts = df.groupby("customer_id")["order_id"].transform("count")
    df["customer_order_count"] = order_counts
    df["is_repeat_customer"] = (order_counts > 1).astype(int)

    avg_spend = df.groupby("customer_id")["cost_of_the_order"].transform("mean")
    df["customer_avg_spend"] = avg_spend.round(2)

    return df


def encode_categoricals(
    df: pd.DataFrame, top_n_restaurants: int = 15
) -> pd.DataFrame:
    """Encode categorical variables for modeling.

    Groups restaurants outside the top N into 'Other' to avoid
    high cardinality, then one-hot encodes all categoricals.

    Args:
        df: DataFrame with categorical columns.
        top_n_restaurants: Number of top restaurants to keep as individual
            categories; the rest become 'Other'.

    Returns:
        DataFrame with encoded features.
    """
    df = df.copy()

    top_restaurants = (
        df["restaurant_name"]
        .value_counts()
        .head(top_n_restaurants)
        .index
    )
    df["restaurant_group"] = np.where(
        df["restaurant_name"].isin(top_restaurants),
        df["restaurant_name"],
        "Other",
    )

    encode_cols = CATEGORICAL_COLS + ["restaurant_group"]
    df = pd.get_dummies(df, columns=encode_cols, drop_first=False, dtype=int)

    return df
