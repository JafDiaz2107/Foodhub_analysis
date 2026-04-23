"""Visualization functions for FoodHub order analysis.

All functions return matplotlib Figure objects for flexible display/saving.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

STYLE_CONFIG = {
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
}

PALETTE = sns.color_palette("Set2")


def _apply_style() -> None:
    plt.rcParams.update(STYLE_CONFIG)
    plt.style.use("seaborn-v0_8-whitegrid")


# --- Data Overview ---


def plot_missing_values(df: pd.DataFrame, figsize: tuple = (10, 5)) -> Figure:
    """Bar chart of missing/NaN percentage per column."""
    _apply_style()
    pct_missing = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    pct_missing = pct_missing[pct_missing > 0]

    fig, ax = plt.subplots(figsize=figsize)
    if pct_missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
    else:
        pct_missing.plot.barh(ax=ax, color=PALETTE[1])
        ax.set_xlabel("% Missing")
        ax.set_title("Missing Values by Column")
        for i, v in enumerate(pct_missing):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    plt.tight_layout()
    return fig


# --- Univariate EDA ---


def plot_numeric_distribution(
    df: pd.DataFrame, column: str, figsize: tuple = (12, 5)
) -> Figure:
    """Histogram with KDE and boxplot for a numeric column."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    sns.histplot(df[column], kde=True, ax=ax1, color=PALETTE[0])
    ax1.set_title(f"Distribution of {column}")

    sns.boxplot(x=df[column], ax=ax2, color=PALETTE[0])
    ax2.set_title(f"Boxplot of {column}")

    plt.tight_layout()
    return fig


def plot_categorical_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int | None = None,
    figsize: tuple = (10, 6),
) -> Figure:
    """Horizontal bar chart of value counts for a categorical column."""
    _apply_style()
    counts = df[column].value_counts()
    if top_n is not None:
        counts = counts.head(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    counts.sort_values().plot.barh(ax=ax, color=PALETTE[2])
    ax.set_xlabel("Count")
    ax.set_title(f"{column} — Value Counts")

    for i, v in enumerate(counts.sort_values()):
        ax.text(v + 0.5, i, str(v), va="center")

    plt.tight_layout()
    return fig


def plot_cuisine_distribution(
    df: pd.DataFrame, figsize: tuple = (8, 8)
) -> Figure:
    """Donut chart of cuisine type proportions."""
    _apply_style()
    counts = df["cuisine_type"].value_counts()

    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=sns.color_palette("Set2", len(counts)),
        pctdistance=0.82, startangle=90,
    )
    centre_circle = plt.Circle((0, 0), 0.60, fc="white")
    ax.add_artist(centre_circle)
    ax.set_title("Cuisine Type Distribution")

    plt.tight_layout()
    return fig


# --- Bivariate EDA ---


def plot_cost_by_cuisine(
    df: pd.DataFrame, figsize: tuple = (12, 6)
) -> Figure:
    """Violin plot of order cost across cuisine types."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    order = df.groupby("cuisine_type")["cost_of_the_order"].median().sort_values().index
    sns.violinplot(
        data=df, x="cuisine_type", y="cost_of_the_order",
        order=order, hue="cuisine_type", palette="Set2", legend=False, ax=ax,
    )
    ax.set_title("Order Cost by Cuisine Type")
    ax.set_xlabel("Cuisine Type")
    ax.set_ylabel("Cost ($)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_time_by_cuisine(
    df: pd.DataFrame, figsize: tuple = (14, 6)
) -> Figure:
    """Side-by-side boxplots of prep time and delivery time by cuisine."""
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    order = (
        df.groupby("cuisine_type")["food_preparation_time"]
        .median()
        .sort_values()
        .index
    )
    sns.boxplot(data=df, x="cuisine_type", y="food_preparation_time",
                order=order, palette="Set2", ax=ax1)
    ax1.set_title("Preparation Time by Cuisine")
    ax1.set_xlabel("")
    ax1.set_ylabel("Minutes")
    ax1.tick_params(axis="x", rotation=45)

    sns.boxplot(data=df, x="cuisine_type", y="delivery_time",
                order=order, palette="Set2", ax=ax2)
    ax2.set_title("Delivery Time by Cuisine")
    ax2.set_xlabel("")
    ax2.set_ylabel("Minutes")
    ax2.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def plot_weekday_weekend_comparison(
    df: pd.DataFrame, figsize: tuple = (12, 5)
) -> Figure:
    """Grouped bar chart comparing key metrics by weekday vs weekend."""
    _apply_style()
    metrics = df.groupby("day_of_the_week").agg(
        avg_cost=("cost_of_the_order", "mean"),
        avg_prep=("food_preparation_time", "mean"),
        avg_delivery=("delivery_time", "mean"),
    ).round(2)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, col, title in zip(
        axes,
        ["avg_cost", "avg_prep", "avg_delivery"],
        ["Avg Cost ($)", "Avg Prep Time (min)", "Avg Delivery Time (min)"],
    ):
        metrics[col].plot.bar(ax=ax, color=[PALETTE[0], PALETTE[1]], rot=0)
        ax.set_title(title)
        ax.set_xlabel("")
        for i, v in enumerate(metrics[col]):
            ax.text(i, v + 0.2, f"{v:.1f}", ha="center")

    plt.tight_layout()
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    method: str = "spearman",
    figsize: tuple = (10, 8),
) -> Figure:
    """Annotated heatmap of correlations among numeric columns."""
    _apply_style()
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(method=method)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, ax=ax,
    )
    ax.set_title(f"Correlation Matrix ({method.title()})")
    plt.tight_layout()
    return fig


def plot_scatter_with_regression(
    df: pd.DataFrame,
    x: str,
    y: str,
    figsize: tuple = (8, 6),
) -> Figure:
    """Scatter plot with regression line for two numeric variables."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(data=df, x=x, y=y, scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title(f"{y} vs {x}")
    plt.tight_layout()
    return fig


# --- Statistical Tests ---


def plot_test_result(
    group_a: pd.Series,
    group_b: pd.Series,
    label_a: str,
    label_b: str,
    metric_name: str,
    p_value: float,
    figsize: tuple = (10, 5),
) -> Figure:
    """Overlaid distributions of two groups with p-value annotation."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.kdeplot(group_a, ax=ax, label=label_a, fill=True, alpha=0.4, color=PALETTE[0])
    sns.kdeplot(group_b, ax=ax, label=label_b, fill=True, alpha=0.4, color=PALETTE[1])

    sig = "significant" if p_value < 0.05 else "not significant"
    ax.set_title(f"{metric_name}: {label_a} vs {label_b} (p={p_value:.4f}, {sig})")
    ax.set_xlabel(metric_name)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_effect_sizes(
    results: dict[str, float], figsize: tuple = (10, 6)
) -> Figure:
    """Horizontal bar chart of effect sizes from multiple tests."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    names = list(results.keys())
    values = list(results.values())
    colors = [PALETTE[0] if v >= 0 else PALETTE[3] for v in values]

    ax.barh(names, values, color=colors)
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Effect Size")
    ax.set_title("Effect Sizes Across Tests")

    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center")

    plt.tight_layout()
    return fig


# --- Customer Segments ---


def plot_segment_sizes(
    segment_counts: pd.Series, figsize: tuple = (10, 6)
) -> Figure:
    """Bar chart of customer segment proportions."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    pcts = (segment_counts / segment_counts.sum() * 100).sort_values()
    pcts.plot.barh(ax=ax, color=sns.color_palette("Set2", len(pcts)))
    ax.set_xlabel("% of Customers")
    ax.set_title("Customer Segment Distribution")

    for i, v in enumerate(pcts):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center")

    plt.tight_layout()
    return fig


def plot_segment_profiles(
    profiles: pd.DataFrame,
    metrics: list[str],
    figsize: tuple = (10, 8),
) -> Figure:
    """Grouped bar chart comparing segments across key metrics (normalized)."""
    _apply_style()
    normalized = profiles[metrics].copy()
    for col in metrics:
        col_max = normalized[col].max()
        if col_max > 0:
            normalized[col] = normalized[col] / col_max

    fig, ax = plt.subplots(figsize=figsize)
    normalized.plot.bar(ax=ax, colormap="Set2")
    ax.set_xticklabels(profiles.index, rotation=45, ha="right")
    ax.set_ylabel("Normalized Value (0-1)")
    ax.set_title("Segment Profiles — Normalized Comparison")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    return fig


def plot_segment_comparison(
    df: pd.DataFrame,
    segment_col: str,
    metric_col: str,
    figsize: tuple = (10, 6),
) -> Figure:
    """Boxplot of a metric across segments."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    order = df.groupby(segment_col)[metric_col].median().sort_values().index
    sns.boxplot(data=df, x=segment_col, y=metric_col,
                order=order, palette="Set2", ax=ax)
    ax.set_title(f"{metric_col} by {segment_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


# --- Predictive Modeling ---


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | None = None,
    figsize: tuple = (7, 6),
) -> Figure:
    """Annotated confusion matrix heatmap."""
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels or "auto",
                yticklabels=labels or "auto")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    return fig


def plot_roc_curve(
    results: dict[str, tuple[np.ndarray, np.ndarray, float]],
    figsize: tuple = (8, 7),
) -> Figure:
    """ROC curves for multiple models.

    Args:
        results: Dict of model_name -> (fpr_array, tpr_array, auc_score).
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, (fpr, tpr, auc)) in enumerate(results.items()):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_feature_importance(
    importances: pd.DataFrame,
    top_n: int = 15,
    figsize: tuple = (10, 7),
) -> Figure:
    """Horizontal bar chart of top feature importances.

    Args:
        importances: DataFrame with 'feature' and 'importance' columns.
        top_n: Number of top features to show.
    """
    _apply_style()
    top = importances.nlargest(top_n, "importance")

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top["feature"], top["importance"], color=PALETTE[0])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig
