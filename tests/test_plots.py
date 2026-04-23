"""Tests for visualization functions."""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib.figure import Figure

from src.visualization.plots import (
    plot_categorical_counts,
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_cost_by_cuisine,
    plot_feature_importance,
    plot_missing_values,
    plot_numeric_distribution,
    plot_segment_sizes,
    plot_test_result,
    plot_weekday_weekend_comparison,
)


class TestPlotFunctions:
    def test_missing_values_returns_figure(self, cleaned_data: pd.DataFrame) -> None:
        fig = plot_missing_values(cleaned_data)
        assert isinstance(fig, Figure)

    def test_numeric_distribution_returns_figure(
        self, cleaned_data: pd.DataFrame
    ) -> None:
        fig = plot_numeric_distribution(cleaned_data, "cost_of_the_order")
        assert isinstance(fig, Figure)

    def test_categorical_counts_returns_figure(
        self, cleaned_data: pd.DataFrame
    ) -> None:
        fig = plot_categorical_counts(cleaned_data, "cuisine_type")
        assert isinstance(fig, Figure)

    def test_cost_by_cuisine_returns_figure(self, cleaned_data: pd.DataFrame) -> None:
        fig = plot_cost_by_cuisine(cleaned_data)
        assert isinstance(fig, Figure)

    def test_correlation_heatmap_returns_figure(
        self, cleaned_data: pd.DataFrame
    ) -> None:
        fig = plot_correlation_heatmap(cleaned_data)
        assert isinstance(fig, Figure)

    def test_weekday_weekend_returns_figure(self, cleaned_data: pd.DataFrame) -> None:
        fig = plot_weekday_weekend_comparison(cleaned_data)
        assert isinstance(fig, Figure)

    def test_confusion_matrix_returns_figure(self) -> None:
        cm = np.array([[40, 10], [5, 45]])
        fig = plot_confusion_matrix(cm, labels=["No", "Yes"])
        assert isinstance(fig, Figure)

    def test_test_result_returns_figure(self) -> None:
        a = pd.Series(np.random.randn(50))
        b = pd.Series(np.random.randn(50) + 1)
        fig = plot_test_result(a, b, "Group A", "Group B", "Value", 0.01)
        assert isinstance(fig, Figure)

    def test_segment_sizes_returns_figure(self) -> None:
        segments = pd.Series({"Loyal": 30, "One-Time": 50, "At-Risk": 20})
        fig = plot_segment_sizes(segments)
        assert isinstance(fig, Figure)

    def test_feature_importance_returns_figure(self) -> None:
        imp = pd.DataFrame({
            "feature": ["cost", "prep_time", "delivery_time"],
            "importance": [0.5, 0.3, 0.2],
        })
        fig = plot_feature_importance(imp, top_n=3)
        assert isinstance(fig, Figure)

    def test_handles_empty_data(self) -> None:
        df = pd.DataFrame({"a": pd.Series(dtype=float)})
        fig = plot_missing_values(df)
        assert isinstance(fig, Figure)
