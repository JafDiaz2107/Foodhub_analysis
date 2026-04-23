"""FoodHub Analysis — Interactive Dashboard."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis.segments import compute_customer_metrics, create_segments, profile_segments
from src.analysis.stats import (
    compare_across_groups,
    compare_weekday_weekend,
    compute_correlation_matrix,
    run_all_tests,
)
from src.data.load import load_foodhub
from src.features.build import clean_data, encode_categoricals, engineer_features
from src.models.predict import compare_models, get_feature_importances, prepare_classification_data
from src.visualization.plots import (
    plot_categorical_counts,
    plot_correlation_heatmap,
    plot_cost_by_cuisine,
    plot_cuisine_distribution,
    plot_missing_values,
    plot_numeric_distribution,
    plot_scatter_with_regression,
    plot_segment_profiles,
    plot_segment_sizes,
    plot_weekday_weekend_comparison,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "foodhub_order.csv"

st.set_page_config(page_title="FoodHub Analysis", page_icon="🍔", layout="wide")
st.title("FoodHub Order Analysis")


@st.cache_data
def load_data():
    df = load_foodhub(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)
    return df


df = load_data()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Explore", "Segments", "Models"])

# --- Tab 1: Overview ---
with tab1:
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df):,}")
    col2.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    col3.metric("Restaurants", f"{df['restaurant_name'].nunique():,}")
    col4.metric("Cuisines", f"{df['cuisine_type'].nunique():,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Avg Order Cost", f"${df['cost_of_the_order'].mean():.2f}")
    col6.metric("Avg Prep Time", f"{df['food_preparation_time'].mean():.0f} min")
    col7.metric("Avg Delivery Time", f"{df['delivery_time'].mean():.0f} min")
    col8.metric("Rating Coverage", f"{df['has_rating'].mean():.0%}")

    st.subheader("Missing Values")
    fig = plot_missing_values(df)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Cuisine Distribution")
    fig = plot_cuisine_distribution(df)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 15 Restaurants")
    fig = plot_categorical_counts(df, "restaurant_name", top_n=15)
    st.pyplot(fig)
    plt.close(fig)

# --- Tab 2: Explore ---
with tab2:
    st.header("Interactive Exploration")

    st.sidebar.header("Filters")
    cuisines = st.sidebar.multiselect(
        "Cuisine Type",
        options=sorted(df["cuisine_type"].unique()),
        default=sorted(df["cuisine_type"].unique()),
    )
    day_filter = st.sidebar.radio(
        "Day of Week", ["All", "Weekday", "Weekend"]
    )

    filtered = df[df["cuisine_type"].isin(cuisines)]
    if day_filter != "All":
        filtered = filtered[filtered["day_of_the_week"] == day_filter]

    st.caption(f"Showing {len(filtered):,} of {len(df):,} orders")

    numeric_cols = [
        "cost_of_the_order", "food_preparation_time", "delivery_time",
        "total_time", "cost_per_minute", "prep_delivery_ratio",
    ]

    col_left, col_right = st.columns(2)
    with col_left:
        selected_col = st.selectbox("Distribution of:", numeric_cols)
        fig = plot_numeric_distribution(filtered, selected_col)
        st.pyplot(fig)
        plt.close(fig)

    with col_right:
        st.subheader("Weekday vs Weekend")
        fig = plot_weekday_weekend_comparison(filtered)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Cost by Cuisine")
    fig = plot_cost_by_cuisine(filtered)
    st.pyplot(fig)
    plt.close(fig)

    col_scatter_x = st.selectbox("Scatter X:", numeric_cols, index=1)
    col_scatter_y = st.selectbox("Scatter Y:", numeric_cols, index=0)
    fig = plot_scatter_with_regression(filtered, col_scatter_x, col_scatter_y)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Correlation Matrix")
    fig = plot_correlation_heatmap(filtered)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Statistical Tests")
    results = run_all_tests(filtered)
    results_df = pd.DataFrame(results)[["name", "test", "statistic", "p_value", "effect_size", "interpretation"]]
    st.dataframe(results_df, use_container_width=True)

# --- Tab 3: Segments ---
with tab3:
    st.header("Customer Segmentation")
    st.caption("Rule-based FMS (Frequency-Monetary-Satisfaction) segmentation")

    metrics = compute_customer_metrics(df)
    segmented = create_segments(metrics)
    profiles = profile_segments(segmented)

    st.subheader("Segment Distribution")
    segment_counts = segmented["segment"].value_counts()
    fig = plot_segment_sizes(segment_counts)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Segment Profiles")
    st.dataframe(profiles, use_container_width=True)

    profile_metrics = ["avg_orders", "avg_spend", "avg_prep_time", "avg_delivery_time"]
    available_metrics = [m for m in profile_metrics if m in profiles.columns]
    if available_metrics:
        fig = plot_segment_profiles(profiles, available_metrics)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Segment Details")
    selected_segment = st.selectbox("Select segment:", sorted(segmented["segment"].unique()))
    segment_data = segmented[segmented["segment"] == selected_segment]
    st.write(f"**{len(segment_data)} customers** in this segment")
    st.dataframe(
        segment_data[["customer_id", "order_count", "avg_spend", "avg_rating",
                       "preferred_cuisine", "frequency_tier", "monetary_tier",
                       "satisfaction_tier"]].head(20),
        use_container_width=True,
    )

# --- Tab 4: Models ---
with tab4:
    st.header("Predictive Modeling")
    st.caption("Binary classification: will a customer leave a rating?")

    df_encoded = encode_categoricals(df)
    X, y = prepare_classification_data(df_encoded)

    st.subheader("Class Balance")
    col1, col2 = st.columns(2)
    col1.metric("Rated (1)", f"{y.sum():,} ({y.mean():.1%})")
    col2.metric("Not Rated (0)", f"{(~y.astype(bool)).sum():,} ({1-y.mean():.1%})")

    st.subheader("Model Comparison (5-Fold Stratified CV)")
    with st.spinner("Training models..."):
        comparison = compare_models(X, y)

    display_cols = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "roc_auc_mean"]
    st.dataframe(
        comparison[display_cols].style.format("{:.3f}"),
        use_container_width=True,
    )

    st.subheader("Feature Importances (Random Forest)")
    importances = get_feature_importances(X, y, model_name="Random Forest")
    from src.visualization.plots import plot_feature_importance
    fig = plot_feature_importance(importances, top_n=15)
    st.pyplot(fig)
    plt.close(fig)
