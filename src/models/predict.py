"""Predictive modeling for FoodHub order analysis.

Primary task: binary classification (rated vs not rated).
Secondary task: order cost regression.
Uses stratified cross-validation for honest evaluation on small data.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
}

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}


def prepare_classification_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target for rating classification.

    Args:
        df: Engineered DataFrame (after feature engineering + encoding).

    Returns:
        Tuple of (X feature matrix, y target series).
    """
    drop_cols = [
        "order_id", "customer_id", "restaurant_name", "rating",
        "has_rating",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    y = df["has_rating"]

    return X, y


def evaluate_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Random Forest",
    n_splits: int = 5,
) -> dict:
    """Evaluate a classifier using stratified k-fold cross-validation.

    Args:
        X: Feature matrix.
        y: Binary target.
        model_name: Key from CLASSIFIERS dict.
        n_splits: Number of CV folds.

    Returns:
        Dict with mean and std of each metric.
    """
    model = CLASSIFIERS[model_name]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = cross_validate(
        model, X, y, cv=cv, scoring=SCORING, return_train_score=False
    )

    return {
        "model": model_name,
        "accuracy_mean": float(np.mean(results["test_accuracy"])),
        "accuracy_std": float(np.std(results["test_accuracy"])),
        "precision_mean": float(np.mean(results["test_precision"])),
        "recall_mean": float(np.mean(results["test_recall"])),
        "f1_mean": float(np.mean(results["test_f1"])),
        "f1_std": float(np.std(results["test_f1"])),
        "roc_auc_mean": float(np.mean(results["test_roc_auc"])),
        "roc_auc_std": float(np.std(results["test_roc_auc"])),
    }


def compare_models(
    X: pd.DataFrame, y: pd.Series
) -> pd.DataFrame:
    """Compare all classifiers via cross-validation.

    Args:
        X: Feature matrix.
        y: Binary target.

    Returns:
        DataFrame with one row per model and metric columns.
    """
    results = []
    for name in CLASSIFIERS:
        result = evaluate_classifier(X, y, model_name=name)
        results.append(result)
    return pd.DataFrame(results).set_index("model")


def get_feature_importances(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Random Forest",
) -> pd.DataFrame:
    """Train a model and extract feature importances.

    Args:
        X: Feature matrix.
        y: Target.
        model_name: Must be a tree-based model with feature_importances_.

    Returns:
        DataFrame with 'feature' and 'importance' columns, sorted descending.
    """
    model = CLASSIFIERS[model_name]
    model.fit(X, y)

    importances = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return importances


def get_roc_data(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "Random Forest",
) -> tuple[np.ndarray, np.ndarray, float]:
    """Train model on full data and compute ROC curve.

    For visualization only — not for model evaluation (use CV for that).

    Args:
        X: Feature matrix.
        y: Target.
        model_name: Key from CLASSIFIERS.

    Returns:
        Tuple of (fpr, tpr, auc_score).
    """
    model = CLASSIFIERS[model_name]
    model.fit(X, y)
    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_proba)
    auc = roc_auc_score(y, y_proba)
    return fpr, tpr, float(auc)
