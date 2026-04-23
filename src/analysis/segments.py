"""Rule-based customer segmentation using Frequency-Monetary-Satisfaction tiers.

Uses behavioral thresholds (not clustering) to segment customers into
actionable groups. Complements the unsupervised approach in the
Customer Segmentation project by showing a different analytical technique.
"""

import numpy as np
import pandas as pd


def compute_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order-level data to customer-level metrics.

    Args:
        df: Order-level DataFrame (after cleaning, with has_rating column).

    Returns:
        DataFrame with one row per customer and aggregated metrics.
    """
    metrics = df.groupby("customer_id").agg(
        order_count=("order_id", "count"),
        total_spend=("cost_of_the_order", "sum"),
        avg_spend=("cost_of_the_order", "mean"),
        avg_prep_time=("food_preparation_time", "mean"),
        avg_delivery_time=("delivery_time", "mean"),
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count"),
        weekend_orders=("day_of_the_week", lambda x: (x == "Weekend").sum()),
    ).reset_index()

    metrics["total_spend"] = metrics["total_spend"].round(2)
    metrics["avg_spend"] = metrics["avg_spend"].round(2)
    metrics["avg_prep_time"] = metrics["avg_prep_time"].round(1)
    metrics["avg_delivery_time"] = metrics["avg_delivery_time"].round(1)
    metrics["weekend_pct"] = (
        metrics["weekend_orders"] / metrics["order_count"] * 100
    ).round(1)

    preferred = df.groupby("customer_id")["cuisine_type"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"
    )
    metrics["preferred_cuisine"] = metrics["customer_id"].map(preferred)

    return metrics


def assign_frequency_tier(
    metrics: pd.DataFrame, thresholds: tuple[int, int] = (1, 3)
) -> pd.Series:
    """Assign frequency tier based on order count.

    Args:
        metrics: Customer-level DataFrame from compute_customer_metrics.
        thresholds: (low_max, high_min) boundaries. Customers with
            order_count <= low_max are Low, >= high_min are High.

    Returns:
        Series of tier labels.
    """
    low_max, high_min = thresholds
    conditions = [
        metrics["order_count"] <= low_max,
        metrics["order_count"] >= high_min,
    ]
    choices = ["Low", "High"]
    return pd.Series(
        np.select(conditions, choices, default="Medium"),
        index=metrics.index,
        name="frequency_tier",
    )


def assign_monetary_tier(
    metrics: pd.DataFrame, n_tiers: int = 3
) -> pd.Series:
    """Assign monetary tier based on average spend using quantiles.

    Args:
        metrics: Customer-level DataFrame.
        n_tiers: Number of tiers (default 3: Low/Medium/High).

    Returns:
        Series of tier labels.
    """
    labels = ["Low", "Medium", "High"][:n_tiers]
    return pd.qcut(
        metrics["avg_spend"], q=n_tiers, labels=labels, duplicates="drop"
    ).rename("monetary_tier")


def assign_satisfaction_tier(metrics: pd.DataFrame) -> pd.Series:
    """Assign satisfaction tier based on average rating.

    Customers with no ratings get 'Unknown'.

    Args:
        metrics: Customer-level DataFrame.

    Returns:
        Series of tier labels.
    """
    conditions = [
        metrics["avg_rating"].isna(),
        metrics["avg_rating"] >= 4,
        metrics["avg_rating"] >= 3,
    ]
    choices = ["Unknown", "Promoter", "Neutral"]
    return pd.Series(
        np.select(conditions, choices, default="Detractor"),
        index=metrics.index,
        name="satisfaction_tier",
    )


def create_segments(metrics: pd.DataFrame) -> pd.DataFrame:
    """Assign all tiers and derive named segments.

    Args:
        metrics: Customer-level DataFrame from compute_customer_metrics.

    Returns:
        DataFrame with tier columns and a final 'segment' column.
    """
    df = metrics.copy()
    df["frequency_tier"] = assign_frequency_tier(df)
    df["monetary_tier"] = assign_monetary_tier(df)
    df["satisfaction_tier"] = assign_satisfaction_tier(df)

    repeat = df["frequency_tier"].isin(["Medium", "High"])
    conditions = [
        (df["frequency_tier"] == "Low"),
        repeat & (df["monetary_tier"] == "High"),
        repeat & (df["satisfaction_tier"] == "Detractor"),
        repeat & (df["monetary_tier"] != "High"),
    ]
    choices = ["One-Time", "Loyal High-Spender", "At-Risk", "Regular"]
    df["segment"] = np.select(conditions, choices, default="Other")

    return df


def profile_segments(segmented: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per segment.

    Args:
        segmented: DataFrame from create_segments.

    Returns:
        DataFrame with one row per segment and profile metrics.
    """
    return segmented.groupby("segment").agg(
        count=("customer_id", "count"),
        avg_orders=("order_count", "mean"),
        avg_spend=("avg_spend", "mean"),
        avg_prep_time=("avg_prep_time", "mean"),
        avg_delivery_time=("avg_delivery_time", "mean"),
        avg_rating=("avg_rating", "mean"),
        weekend_pct=("weekend_pct", "mean"),
    ).round(2)
