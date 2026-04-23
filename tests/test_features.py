"""Tests for feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.build import clean_data, encode_categoricals, engineer_features


class TestCleanData:
    def test_adds_has_rating_flag(self, cleaned_data: pd.DataFrame) -> None:
        result = clean_data(cleaned_data)
        assert "has_rating" in result.columns
        assert result["has_rating"].dtype == int
        assert set(result["has_rating"].unique()).issubset({0, 1})

    def test_has_rating_matches_nan(self, cleaned_data: pd.DataFrame) -> None:
        result = clean_data(cleaned_data)
        assert (result["has_rating"] == result["rating"].notna().astype(int)).all()

    def test_negative_cost_raises(self, cleaned_data: pd.DataFrame) -> None:
        df = cleaned_data.copy()
        df.loc[0, "cost_of_the_order"] = -5.0
        with pytest.raises(ValueError, match="Negative order costs"):
            clean_data(df)

    def test_negative_prep_time_raises(self, cleaned_data: pd.DataFrame) -> None:
        df = cleaned_data.copy()
        df.loc[0, "food_preparation_time"] = -1
        with pytest.raises(ValueError, match="Negative preparation times"):
            clean_data(df)


class TestEngineerFeatures:
    def test_total_time_computed(self, cleaned_data: pd.DataFrame) -> None:
        result = engineer_features(clean_data(cleaned_data))
        expected = cleaned_data["food_preparation_time"] + cleaned_data["delivery_time"]
        pd.testing.assert_series_equal(
            result["total_time"], expected, check_names=False
        )

    def test_cost_per_minute_positive(self, cleaned_data: pd.DataFrame) -> None:
        result = engineer_features(clean_data(cleaned_data))
        assert (result["cost_per_minute"] >= 0).all()

    def test_prep_delivery_ratio_handles_zero(self) -> None:
        df = pd.DataFrame({
            "order_id": [1],
            "customer_id": [100],
            "restaurant_name": ["Test"],
            "cuisine_type": ["American"],
            "cost_of_the_order": [10.0],
            "day_of_the_week": ["Weekday"],
            "rating": [np.nan],
            "food_preparation_time": [20],
            "delivery_time": [0],
            "has_rating": [0],
        })
        result = engineer_features(df)
        assert result["prep_delivery_ratio"].iloc[0] == 0.0

    def test_repeat_customer_detection(self, cleaned_data: pd.DataFrame) -> None:
        result = engineer_features(clean_data(cleaned_data))
        for cid, group in result.groupby("customer_id"):
            if len(group) > 1:
                assert (group["is_repeat_customer"] == 1).all()
            else:
                assert (group["is_repeat_customer"] == 0).all()

    def test_is_weekend_flag(self, cleaned_data: pd.DataFrame) -> None:
        result = engineer_features(clean_data(cleaned_data))
        weekend_mask = result["day_of_the_week"] == "Weekend"
        assert (result.loc[weekend_mask, "is_weekend"] == 1).all()
        assert (result.loc[~weekend_mask, "is_weekend"] == 0).all()


class TestEncodeCategoricals:
    def test_creates_dummy_columns(self, cleaned_data: pd.DataFrame) -> None:
        df = engineer_features(clean_data(cleaned_data))
        result = encode_categoricals(df)
        assert any(col.startswith("cuisine_type_") for col in result.columns)
        assert any(col.startswith("day_of_the_week_") for col in result.columns)

    def test_restaurant_grouping(self, cleaned_data: pd.DataFrame) -> None:
        df = engineer_features(clean_data(cleaned_data))
        result = encode_categoricals(df, top_n_restaurants=2)
        restaurant_cols = [
            c for c in result.columns
            if c.startswith("restaurant_group_")
        ]
        assert any("Other" in c for c in restaurant_cols)

    def test_no_original_categorical_columns(self, cleaned_data: pd.DataFrame) -> None:
        df = engineer_features(clean_data(cleaned_data))
        result = encode_categoricals(df)
        assert "cuisine_type" not in result.columns
        assert "day_of_the_week" not in result.columns
