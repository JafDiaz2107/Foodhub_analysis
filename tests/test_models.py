"""Tests for predictive modeling."""

import pandas as pd
import pytest

from src.features.build import clean_data, encode_categoricals, engineer_features
from src.models.predict import (
    compare_models,
    evaluate_classifier,
    get_feature_importances,
    get_roc_data,
    prepare_classification_data,
)


@pytest.fixture
def model_data(cleaned_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = clean_data(cleaned_data)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return prepare_classification_data(df)


class TestPrepareData:
    def test_returns_x_and_y(self, model_data: tuple) -> None:
        X, y = model_data
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_is_binary(self, model_data: tuple) -> None:
        _, y = model_data
        assert set(y.unique()).issubset({0, 1})

    def test_no_target_in_features(self, model_data: tuple) -> None:
        X, _ = model_data
        assert "has_rating" not in X.columns
        assert "rating" not in X.columns


class TestEvaluation:
    def test_returns_expected_keys(self, model_data: tuple) -> None:
        X, y = model_data
        result = evaluate_classifier(X, y, model_name="Logistic Regression")
        assert "accuracy_mean" in result
        assert "f1_mean" in result
        assert "roc_auc_mean" in result

    def test_metrics_in_valid_range(self, model_data: tuple) -> None:
        X, y = model_data
        result = evaluate_classifier(X, y, model_name="Logistic Regression")
        assert 0 <= result["accuracy_mean"] <= 1
        assert 0 <= result["f1_mean"] <= 1
        assert 0 <= result["roc_auc_mean"] <= 1

    def test_compare_returns_all_models(self, model_data: tuple) -> None:
        X, y = model_data
        comparison = compare_models(X, y)
        assert len(comparison) == 3
        assert "Logistic Regression" in comparison.index

    def test_feature_importances_sum(self, model_data: tuple) -> None:
        X, y = model_data
        imp = get_feature_importances(X, y, model_name="Random Forest")
        assert abs(imp["importance"].sum() - 1.0) < 0.01

    def test_roc_data_shape(self, model_data: tuple) -> None:
        X, y = model_data
        fpr, tpr, auc = get_roc_data(X, y)
        assert len(fpr) == len(tpr)
        assert 0 <= auc <= 1
