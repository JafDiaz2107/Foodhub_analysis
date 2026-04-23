"""Tests for statistical analysis functions."""

import numpy as np
import pandas as pd

from src.analysis.stats import (
    compare_across_groups,
    compare_independence,
    compare_two_groups,
    compute_correlation_matrix,
    run_all_tests,
)
from src.features.build import clean_data

REQUIRED_KEYS = {
    "test", "name", "statistic",
    "p_value", "effect_size", "interpretation",
}


class TestTwoGroups:
    def test_returns_required_keys(self) -> None:
        a = pd.Series(np.random.randn(30))
        b = pd.Series(np.random.randn(30) + 1)
        result = compare_two_groups(a, b, name="test")
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_p_value_in_range(self) -> None:
        a = pd.Series(np.random.randn(30))
        b = pd.Series(np.random.randn(30))
        result = compare_two_groups(a, b)
        assert 0 <= result["p_value"] <= 1

    def test_handles_nan(self) -> None:
        a = pd.Series([1, 2, np.nan, 4, 5])
        b = pd.Series([6, np.nan, 8, 9, 10])
        result = compare_two_groups(a, b)
        assert result["n_a"] == 4
        assert result["n_b"] == 4


class TestIndependence:
    def test_returns_required_keys(self, cleaned_data: pd.DataFrame) -> None:
        result = compare_independence(cleaned_data, "cuisine_type", "day_of_the_week")
        expected = REQUIRED_KEYS
        assert expected.issubset(result.keys())

    def test_cramers_v_nonnegative(self, cleaned_data: pd.DataFrame) -> None:
        result = compare_independence(cleaned_data, "cuisine_type", "day_of_the_week")
        assert result["effect_size"] >= 0


class TestAcrossGroups:
    def test_returns_required_keys(
        self, cleaned_data: pd.DataFrame
    ) -> None:
        result = compare_across_groups(
            cleaned_data, "cuisine_type", "cost_of_the_order"
        )
        assert REQUIRED_KEYS.issubset(result.keys())

    def test_n_groups_correct(
        self, cleaned_data: pd.DataFrame
    ) -> None:
        result = compare_across_groups(
            cleaned_data, "cuisine_type", "cost_of_the_order"
        )
        assert result["n_groups"] == cleaned_data["cuisine_type"].nunique()


class TestCorrelationMatrix:
    def test_is_square(self, cleaned_data: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(cleaned_data)
        assert corr.shape[0] == corr.shape[1]

    def test_diagonal_is_one(self, cleaned_data: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(cleaned_data)
        np.testing.assert_array_almost_equal(np.diag(corr.values), 1.0)

    def test_symmetric(self, cleaned_data: pd.DataFrame) -> None:
        corr = compute_correlation_matrix(cleaned_data)
        pd.testing.assert_frame_equal(corr, corr.T)


class TestRunAllTests:
    def test_returns_list(self, cleaned_data: pd.DataFrame) -> None:
        df = clean_data(cleaned_data)
        results = run_all_tests(df)
        assert isinstance(results, list)
        assert len(results) >= 5
