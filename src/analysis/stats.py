"""Statistical hypothesis tests for FoodHub order analysis.

Uses non-parametric tests throughout — appropriate for small, skewed data.
Each test function returns a standardized result dict with:
statistic, p_value, effect_size, and interpretation.
"""

import numpy as np
import pandas as pd
from scipy import stats


def _interpret_p(p_value: float, alpha: float = 0.05) -> str:
    return "significant" if p_value < alpha else "not significant"


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    return 1 - (2 * u_stat) / (n1 * n2)


def compare_two_groups(
    group_a: pd.Series,
    group_b: pd.Series,
    name: str = "",
) -> dict:
    """Mann-Whitney U test comparing two independent groups.

    Args:
        group_a: Values from group A.
        group_b: Values from group B.
        name: Label for the test.

    Returns:
        Dict with statistic, p_value, effect_size, interpretation.
    """
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    u_stat, p_value = stats.mannwhitneyu(
        group_a, group_b, alternative="two-sided"
    )
    effect = _rank_biserial(u_stat, len(group_a), len(group_b))

    return {
        "test": "Mann-Whitney U",
        "name": name,
        "statistic": float(u_stat),
        "p_value": float(p_value),
        "effect_size": float(effect),
        "interpretation": _interpret_p(p_value),
        "n_a": len(group_a),
        "n_b": len(group_b),
    }


def compare_weekday_weekend(df: pd.DataFrame, column: str) -> dict:
    """Compare a numeric column between weekday and weekend orders."""
    weekday = df.loc[df["day_of_the_week"] == "Weekday", column]
    weekend = df.loc[df["day_of_the_week"] == "Weekend", column]
    return compare_two_groups(weekday, weekend, name=f"{column}: Weekday vs Weekend")


def compare_independence(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
) -> dict:
    """Chi-square test of independence between two categorical columns.

    Args:
        df: DataFrame.
        col_a: First categorical column.
        col_b: Second categorical column.

    Returns:
        Dict with statistic, p_value, effect_size (Cramér's V), interpretation.
    """
    contingency = pd.crosstab(df[col_a], df[col_b])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    n = contingency.sum().sum()
    k = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

    return {
        "test": "Chi-square",
        "name": f"{col_a} vs {col_b}",
        "statistic": float(chi2),
        "p_value": float(p_value),
        "effect_size": float(cramers_v),
        "dof": int(dof),
        "interpretation": _interpret_p(p_value),
    }


def compare_across_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> dict:
    """Kruskal-Wallis test comparing a metric across 3+ groups.

    Args:
        df: DataFrame.
        group_col: Categorical column defining groups.
        value_col: Numeric column to compare.

    Returns:
        Dict with statistic, p_value, effect_size (eta-squared), interpretation.
    """
    groups = [
        group[value_col].dropna().values
        for _, group in df.groupby(group_col)
        if len(group[value_col].dropna()) > 0
    ]

    h_stat, p_value = stats.kruskal(*groups)

    n = sum(len(g) for g in groups)
    k = len(groups)
    eta_squared = (h_stat - k + 1) / (n - k) if n > k else 0.0

    return {
        "test": "Kruskal-Wallis",
        "name": f"{value_col} across {group_col}",
        "statistic": float(h_stat),
        "p_value": float(p_value),
        "effect_size": float(eta_squared),
        "n_groups": k,
        "interpretation": _interpret_p(p_value),
    }


def compute_correlation_matrix(
    df: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    """Compute correlation matrix for numeric columns.

    Args:
        df: DataFrame.
        method: Correlation method ("spearman" or "pearson").

    Returns:
        Square correlation DataFrame.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr(method=method)


def run_all_tests(df: pd.DataFrame) -> list[dict]:
    """Run the full battery of hypothesis tests on FoodHub data.

    Returns:
        List of result dicts from all tests.
    """
    results = []

    results.append(compare_weekday_weekend(df, "cost_of_the_order"))
    results.append(compare_weekday_weekend(df, "delivery_time"))
    results.append(compare_weekday_weekend(df, "food_preparation_time"))

    results.append(compare_independence(df, "cuisine_type", "day_of_the_week"))

    if "has_rating" in df.columns:
        results.append(compare_independence(df, "day_of_the_week", "has_rating"))

    results.append(compare_across_groups(df, "cuisine_type", "cost_of_the_order"))
    results.append(compare_across_groups(df, "cuisine_type", "delivery_time"))

    return results
