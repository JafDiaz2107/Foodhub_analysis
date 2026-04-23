"""Shared pytest fixtures for FoodHub analysis tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_foodhub_data() -> pd.DataFrame:
    """Generate a synthetic DataFrame matching the raw FoodHub CSV schema.

    Includes 'Not given' strings in the rating column to test loading logic.
    """
    np.random.seed(42)
    n = 50
    restaurants = [
        "Shake Shack", "Blue Ribbon Sushi", "Cafe Habana",
        "Hangawi", "Red Rooster", "Taco Bell",
    ]
    cuisines = ["American", "Japanese", "Mexican", "Korean", "Italian"]

    ratings = np.random.choice(
        ["3", "4", "5", "Not given"], n, p=[0.15, 0.20, 0.26, 0.39]
    )

    return pd.DataFrame({
        "order_id": range(1000, 1000 + n),
        "customer_id": np.random.choice(range(100, 130), n),
        "restaurant_name": np.random.choice(restaurants, n),
        "cuisine_type": np.random.choice(cuisines, n),
        "cost_of_the_order": np.round(np.random.uniform(5.0, 40.0, n), 2),
        "day_of_the_week": np.random.choice(["Weekday", "Weekend"], n, p=[0.7, 0.3]),
        "rating": ratings,
        "food_preparation_time": np.random.randint(15, 45, n),
        "delivery_time": np.random.randint(10, 35, n),
    })


@pytest.fixture
def cleaned_data(sample_foodhub_data: pd.DataFrame) -> pd.DataFrame:
    """Return sample data with rating converted to float (NaN for 'Not given')."""
    df = sample_foodhub_data.copy()
    df["rating"] = df["rating"].replace("Not given", np.nan).astype(float)
    return df
