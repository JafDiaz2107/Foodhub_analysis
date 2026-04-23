"""Tests for data loading and schema validation."""

import pandas as pd
import pytest

from src.data.load import EXPECTED_COLUMNS, load_foodhub, validate_schema


class TestValidateSchema:
    def test_valid_schema_passes(self, sample_foodhub_data: pd.DataFrame) -> None:
        assert validate_schema(sample_foodhub_data) is True

    def test_missing_column_raises(self, sample_foodhub_data: pd.DataFrame) -> None:
        df = sample_foodhub_data.drop(columns=["rating"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_extra_columns_accepted(self, sample_foodhub_data: pd.DataFrame) -> None:
        df = sample_foodhub_data.copy()
        df["extra_col"] = 1
        assert validate_schema(df) is True


class TestLoadFoodhub:
    def test_load_returns_dataframe(self, tmp_path: object) -> None:
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "order_id": [1],
            "customer_id": [100],
            "restaurant_name": ["Test"],
            "cuisine_type": ["American"],
            "cost_of_the_order": [10.0],
            "day_of_the_week": ["Weekday"],
            "rating": ["5"],
            "food_preparation_time": [20],
            "delivery_time": [15],
        })
        df.to_csv(csv_path, index=False)
        result = load_foodhub(csv_path)
        assert isinstance(result, pd.DataFrame)
        assert set(EXPECTED_COLUMNS).issubset(set(result.columns))

    def test_rating_not_given_becomes_nan(self, tmp_path: object) -> None:
        csv_path = tmp_path / "test.csv"
        df = pd.DataFrame({
            "order_id": [1, 2],
            "customer_id": [100, 101],
            "restaurant_name": ["A", "B"],
            "cuisine_type": ["American", "Mexican"],
            "cost_of_the_order": [10.0, 15.0],
            "day_of_the_week": ["Weekday", "Weekend"],
            "rating": ["5", "Not given"],
            "food_preparation_time": [20, 25],
            "delivery_time": [15, 20],
        })
        df.to_csv(csv_path, index=False)
        result = load_foodhub(csv_path)
        assert result["rating"].dtype == float
        assert result["rating"].isna().sum() == 1
        assert result["rating"].iloc[0] == 5.0

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            load_foodhub("/nonexistent/path.csv")
