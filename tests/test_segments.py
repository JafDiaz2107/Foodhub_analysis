"""Tests for customer segmentation."""

import numpy as np
import pandas as pd

from src.analysis.segments import (
    assign_frequency_tier,
    assign_satisfaction_tier,
    compute_customer_metrics,
    create_segments,
    profile_segments,
)
from src.features.build import clean_data


class TestCustomerMetrics:
    def test_one_row_per_customer(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        assert metrics["customer_id"].is_unique

    def test_order_count_positive(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        assert (metrics["order_count"] > 0).all()

    def test_has_expected_columns(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        expected = {"customer_id", "order_count", "total_spend", "avg_spend",
                    "preferred_cuisine", "weekend_pct"}
        assert expected.issubset(set(metrics.columns))


class TestTierAssignment:
    def test_frequency_covers_all(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        tiers = assign_frequency_tier(metrics)
        assert len(tiers) == len(metrics)
        assert not tiers.isna().any()

    def test_frequency_monotonic(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        tiers = assign_frequency_tier(metrics)
        low_mask = tiers == "Low"
        high_mask = tiers == "High"
        if low_mask.any() and high_mask.any():
            assert metrics.loc[low_mask, "order_count"].max() <= \
                   metrics.loc[high_mask, "order_count"].min()

    def test_satisfaction_unknown_for_no_rating(self) -> None:
        metrics = pd.DataFrame({
            "customer_id": [1, 2],
            "avg_rating": [np.nan, 5.0],
        })
        tiers = assign_satisfaction_tier(metrics)
        assert tiers.iloc[0] == "Unknown"
        assert tiers.iloc[1] == "Promoter"


class TestCreateSegments:
    def test_no_customer_unsegmented(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        segmented = create_segments(metrics)
        assert not segmented["segment"].isna().any()

    def test_profile_has_expected_structure(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        metrics = compute_customer_metrics(df)
        segmented = create_segments(metrics)
        profiles = profile_segments(segmented)
        assert "count" in profiles.columns
        assert "avg_spend" in profiles.columns
        assert profiles["count"].sum() == len(metrics)
