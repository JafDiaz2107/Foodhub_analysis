"""Generate all analysis notebooks programmatically."""

import nbformat as nbf
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
NOTEBOOKS_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    for cell_type, source in cells:
        if cell_type == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    return nb


# --- Notebook 00: Data Overview ---
nb00 = make_nb([
    ("md", "# 00 — Data Overview\n\nLoad the FoodHub dataset, inspect its structure, check data quality, and understand what we're working with."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.visualization.plots import plot_missing_values"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\nprint(f"Shape: {df.shape}")\ndf.head()'),
    ("md", "## Data Types and Basic Info"),
    ("code", "df.info()"),
    ("code", "df.describe()"),
    ("md", "## Missing Values\n\nThe `rating` column has ~39% NaN values — these were originally recorded as `\"Not given\"` strings and converted to NaN during loading."),
    ("code", "pct_missing = df.isna().mean() * 100\npct_missing[pct_missing > 0]"),
    ("code", "fig = plot_missing_values(df)\nfig"),
    ("md", "## Value Ranges and Uniqueness"),
    ("code", 'print(f"Unique customers: {df[\'customer_id\'].nunique()}")\nprint(f"Unique restaurants: {df[\'restaurant_name\'].nunique()}")\nprint(f"Unique cuisines: {df[\'cuisine_type\'].nunique()}")\nprint(f"Orders: {len(df)}")\nprint(f"\\nDay distribution:\\n{df[\'day_of_the_week\'].value_counts()}")\nprint(f"\\nRating distribution (excluding NaN):\\n{df[\'rating\'].value_counts()}")'),
    ("md", "## Key Observations\n\n- **1,898 orders** from **1,200 unique customers** across **178 restaurants** and **14 cuisines**\n- **39% of orders have no rating** — significant enough to analyze as a pattern, not just missing data\n- No other missing values, no duplicates\n- Cost ranges from ~$4 to ~$36, prep time 20-35 min, delivery 15-33 min"),
])

# --- Notebook 01: Univariate EDA ---
nb01 = make_nb([
    ("md", "# 01 — Univariate EDA\n\nExamine the distribution of each variable individually to understand central tendency, spread, and outliers."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.features.build import clean_data, engineer_features\nfrom src.visualization.plots import (\n    plot_numeric_distribution,\n    plot_categorical_counts,\n    plot_cuisine_distribution,\n)"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\ndf = clean_data(df)\ndf = engineer_features(df)'),
    ("md", "## Order Cost Distribution"),
    ("code", 'fig = plot_numeric_distribution(df, "cost_of_the_order")\nfig'),
    ("md", "Cost is roughly uniform between $5-$35 with no extreme outliers. The median and mean are close, suggesting relatively symmetric distribution."),
    ("md", "## Food Preparation Time"),
    ("code", 'fig = plot_numeric_distribution(df, "food_preparation_time")\nfig'),
    ("md", "Prep time centers around 27 minutes with a range of 20-35 minutes. The distribution is fairly uniform — no strong skew."),
    ("md", "## Delivery Time"),
    ("code", 'fig = plot_numeric_distribution(df, "delivery_time")\nfig'),
    ("md", "Delivery time has a wider range (15-33 min) and shows slight right skew. Some deliveries take notably longer."),
    ("md", "## Total Time (Prep + Delivery)"),
    ("code", 'fig = plot_numeric_distribution(df, "total_time")\nfig'),
    ("md", "## Cuisine Types"),
    ("code", "fig = plot_cuisine_distribution(df)\nfig"),
    ("md", "American and Japanese cuisines dominate, together accounting for over 45% of orders."),
    ("md", "## Top Restaurants"),
    ("code", 'fig = plot_categorical_counts(df, "restaurant_name", top_n=15)\nfig'),
    ("md", "Shake Shack leads with 219 orders (11.5%). The top 10 restaurants account for ~46% of all orders — a strong concentration pattern consistent with power-law distributions in marketplace platforms."),
    ("md", "## Rating Distribution"),
    ("code", 'rated = df[df["rating"].notna()]\nfig = plot_categorical_counts(rated, "rating")\nfig'),
    ("md", "Among orders that received a rating, scores cluster at 4 and 5. Very few orders receive a rating of 3. This ceiling effect is common in food delivery — dissatisfied customers are more likely to skip rating entirely."),
])

# --- Notebook 02: Bivariate EDA ---
nb02 = make_nb([
    ("md", "# 02 — Bivariate EDA\n\nExplore relationships between pairs of variables to find patterns that single-variable analysis can't reveal."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.features.build import clean_data, engineer_features\nfrom src.visualization.plots import (\n    plot_cost_by_cuisine,\n    plot_time_by_cuisine,\n    plot_weekday_weekend_comparison,\n    plot_correlation_heatmap,\n    plot_scatter_with_regression,\n)"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\ndf = clean_data(df)\ndf = engineer_features(df)'),
    ("md", "## Cost by Cuisine"),
    ("code", "fig = plot_cost_by_cuisine(df)\nfig"),
    ("md", "Median cost is fairly consistent across cuisines, but the spread varies. Some cuisines (Korean, French, Italian) show wider price ranges, suggesting both budget and premium options."),
    ("md", "## Prep and Delivery Time by Cuisine"),
    ("code", "fig = plot_time_by_cuisine(df)\nfig"),
    ("md", "Preparation time shows more cuisine-to-cuisine variation than delivery time. This makes sense — delivery depends on distance and logistics, while prep depends on the food itself."),
    ("md", "## Weekday vs Weekend"),
    ("code", "fig = plot_weekday_weekend_comparison(df)\nfig"),
    ("md", "Average cost, prep time, and delivery time are all slightly higher on weekends. The question is whether these differences are statistically meaningful — we'll test that in notebook 03."),
    ("md", "## Prep Time vs Cost"),
    ("code", 'fig = plot_scatter_with_regression(df, "food_preparation_time", "cost_of_the_order")\nfig'),
    ("md", "No clear linear relationship between prep time and order cost. Expensive orders don't necessarily take longer to prepare."),
    ("md", "## Prep Time vs Delivery Time"),
    ("code", 'fig = plot_scatter_with_regression(df, "food_preparation_time", "delivery_time")\nfig'),
    ("md", "No strong correlation between prep and delivery times. These are largely independent processes."),
    ("md", "## Correlation Matrix"),
    ("code", "fig = plot_correlation_heatmap(df)\nfig"),
    ("md", "The Spearman correlation matrix confirms weak correlations between most numeric variables. The strongest relationship is between `total_time` and its components (mechanical — it's a sum). No multicollinearity concerns for modeling."),
])

# --- Notebook 03: Statistical Tests ---
nb03 = make_nb([
    ("md", "# 03 — Statistical Tests\n\nMove beyond visual patterns to test whether observed differences are statistically significant. We use non-parametric tests throughout — appropriate for our skewed, bounded distributions."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.features.build import clean_data, engineer_features\nfrom src.analysis.stats import (\n    compare_weekday_weekend,\n    compare_independence,\n    compare_across_groups,\n    run_all_tests,\n)\nfrom src.visualization.plots import plot_test_result, plot_effect_sizes"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\ndf = clean_data(df)\ndf = engineer_features(df)'),
    ("md", "## Weekday vs Weekend: Order Cost\n\nDo customers spend more on weekends?"),
    ("code", 'result = compare_weekday_weekend(df, "cost_of_the_order")\nfor k, v in result.items():\n    print(f"{k}: {v}")'),
    ("code", 'weekday = df.loc[df["day_of_the_week"] == "Weekday", "cost_of_the_order"]\nweekend = df.loc[df["day_of_the_week"] == "Weekend", "cost_of_the_order"]\nfig = plot_test_result(weekday, weekend, "Weekday", "Weekend", "Order Cost ($)", result["p_value"])\nfig'),
    ("md", "## Weekday vs Weekend: Delivery Time"),
    ("code", 'result = compare_weekday_weekend(df, "delivery_time")\nfor k, v in result.items():\n    print(f"{k}: {v}")'),
    ("md", "## Cuisine Preferences by Day\n\nDo people order different cuisines on weekends vs weekdays? (Chi-square test)"),
    ("code", 'result = compare_independence(df, "cuisine_type", "day_of_the_week")\nfor k, v in result.items():\n    print(f"{k}: {v}")'),
    ("md", "## Rating Behavior by Day\n\nAre customers more likely to leave a rating on weekdays or weekends?"),
    ("code", 'result = compare_independence(df, "day_of_the_week", "has_rating")\nfor k, v in result.items():\n    print(f"{k}: {v}")'),
    ("md", "## Cost Across Cuisines\n\nDo different cuisines have significantly different price points? (Kruskal-Wallis)"),
    ("code", 'result = compare_across_groups(df, "cuisine_type", "cost_of_the_order")\nfor k, v in result.items():\n    print(f"{k}: {v}")'),
    ("md", "## All Tests Summary"),
    ("code", 'results = run_all_tests(df)\nsummary = pd.DataFrame(results)[["name", "test", "p_value", "effect_size", "interpretation"]]\nsummary'),
    ("md", "## Effect Sizes"),
    ("code", 'effect_dict = {r["name"]: abs(r["effect_size"]) for r in results}\nfig = plot_effect_sizes(effect_dict)\nfig'),
    ("md", "## Key Takeaways\n\n- Most weekday vs weekend differences show **small effect sizes** even when statistically significant — practically meaningful differences are limited\n- Cuisine type has the strongest influence on cost (expected) and delivery time\n- Rating behavior is largely independent of day of week\n- The small effect sizes suggest that day-of-week is not a strong driver of operational differences"),
])

# --- Notebook 04: Customer Segments ---
nb04 = make_nb([
    ("md", "# 04 — Customer Segmentation\n\nSegment customers using rule-based Frequency-Monetary-Satisfaction (FMS) tiers. This approach uses interpretable business rules rather than clustering algorithms, producing segments that can be directly actioned by a marketing team."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.features.build import clean_data, engineer_features\nfrom src.analysis.segments import (\n    compute_customer_metrics,\n    create_segments,\n    profile_segments,\n)\nfrom src.visualization.plots import (\n    plot_segment_sizes,\n    plot_segment_profiles,\n    plot_segment_comparison,\n)"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\ndf = clean_data(df)\ndf = engineer_features(df)'),
    ("md", "## Customer-Level Metrics"),
    ("code", "metrics = compute_customer_metrics(df)\nprint(f\"Unique customers: {len(metrics)}\")\nmetrics.describe()"),
    ("code", 'print(f"Repeat customers: {(metrics[\'order_count\'] > 1).sum()} ({(metrics[\'order_count\'] > 1).mean():.1%})")\nprint(f"One-time customers: {(metrics[\'order_count\'] == 1).sum()} ({(metrics[\'order_count\'] == 1).mean():.1%})")'),
    ("md", "**65% of customers ordered only once.** This is the dominant pattern and a key business challenge — retention, not acquisition, is where the leverage is."),
    ("md", "## Segmentation"),
    ("code", "segmented = create_segments(metrics)\nsegmented[['customer_id', 'order_count', 'avg_spend', 'avg_rating',\n           'frequency_tier', 'monetary_tier', 'satisfaction_tier', 'segment']].head(10)"),
    ("md", "## Segment Distribution"),
    ("code", "segment_counts = segmented['segment'].value_counts()\nprint(segment_counts)\nprint(f'\\nPercentages:\\n{(segment_counts / len(segmented) * 100).round(1)}')\n\nfig = plot_segment_sizes(segment_counts)\nfig"),
    ("md", "## Segment Profiles"),
    ("code", "profiles = profile_segments(segmented)\nprofiles"),
    ("code", 'fig = plot_segment_profiles(profiles, ["avg_orders", "avg_spend", "avg_prep_time", "avg_delivery_time"])\nfig'),
    ("md", "## Segment Comparisons"),
    ("code", 'fig = plot_segment_comparison(segmented, "segment", "avg_spend")\nfig'),
    ("md", "## Business Recommendations\n\n| Segment | Action |\n|---|---|\n| **One-Time** | Re-engagement campaigns: first-order discount, personalized cuisine recommendations based on their single order |\n| **Loyal High-Spender** | VIP treatment: priority delivery, exclusive restaurant access, loyalty rewards |\n| **Regular** | Upsell: suggest higher-value items, bundle deals to increase average order value |\n| **At-Risk** | Investigate: what's driving low satisfaction? Target with surveys and service recovery offers |"),
])

# --- Notebook 05: Predictive Modeling ---
nb05 = make_nb([
    ("md", "# 05 — Predictive Modeling\n\nCan we predict whether a customer will leave a rating? This is a binary classification problem with a ~61/39 class split. We compare Logistic Regression, Random Forest, and Gradient Boosting using stratified 5-fold cross-validation."),
    ("code", "from pathlib import Path\nimport pandas as pd\nfrom src.data.load import load_foodhub\nfrom src.features.build import clean_data, engineer_features, encode_categoricals\nfrom src.models.predict import (\n    prepare_classification_data,\n    evaluate_classifier,\n    compare_models,\n    get_feature_importances,\n    get_roc_data,\n)\nfrom src.visualization.plots import (\n    plot_feature_importance,\n    plot_roc_curve,\n)"),
    ("code", 'df = load_foodhub(Path("data/raw/foodhub_order.csv"))\ndf = clean_data(df)\ndf = engineer_features(df)\ndf_encoded = encode_categoricals(df)\nX, y = prepare_classification_data(df_encoded)\nprint(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")\nprint(f"Class balance: {y.mean():.1%} rated, {1-y.mean():.1%} not rated")'),
    ("md", "## Baseline\n\nA naive classifier that always predicts the majority class (rated) achieves ~61% accuracy. Our models need to meaningfully beat this."),
    ("md", "## Model Comparison"),
    ("code", "comparison = compare_models(X, y)\ncomparison"),
    ("md", "## ROC Curves"),
    ("code", 'roc_results = {}\nfor name in ["Logistic Regression", "Random Forest", "Gradient Boosting"]:\n    fpr, tpr, auc = get_roc_data(X, y, model_name=name)\n    roc_results[name] = (fpr, tpr, auc)\n\nfig = plot_roc_curve(roc_results)\nfig'),
    ("md", "## Feature Importances"),
    ("code", 'importances = get_feature_importances(X, y, model_name="Random Forest")\nimportances.head(15)'),
    ("code", "fig = plot_feature_importance(importances, top_n=15)\nfig"),
    ("md", "## Limitations\n\n- **Small dataset** (1,898 rows) limits model complexity and generalizability\n- **No temporal features** — we can't model time-of-day or recency effects\n- **ROC on training data** is shown for visualization only — all evaluation metrics come from cross-validation\n- **Feature engineering is constrained** by the 9 original columns — real-world models would include user history, restaurant metadata, and external signals\n\n## What This Demonstrates\n\nEven with limited data, the modeling pipeline shows:\n1. Proper cross-validation methodology (stratified, not random split)\n2. Multiple model comparison with consistent evaluation\n3. Feature importance analysis for interpretability\n4. Honest reporting of limitations alongside results"),
])

# --- Notebook 06: Final Report ---
nb06 = make_nb([
    ("md", "# 06 — Final Report: FoodHub Order Analysis\n\n## Executive Summary\n\nThis analysis examined 1,898 orders from FoodHub, a food delivery platform aggregating orders from 178 restaurants across 14 cuisines. The goal was to understand demand patterns, operational bottlenecks, and customer behavior to inform business strategy."),
    ("md", "## Key Findings\n\n### 1. Demand Concentration\n- The top 10 restaurants handle **46% of all orders** — platform revenue is heavily dependent on a small number of partners\n- American and Japanese cuisines together account for **45%+ of orders**\n- This concentration creates both opportunity (focus partnership efforts) and risk (partner churn impact)\n\n### 2. Customer Retention is the Core Challenge\n- **65% of customers ordered only once** and never returned\n- Repeat customers spend more per order on average\n- Segmentation identified 4 groups: One-Time (largest), Regular, Loyal High-Spender, and At-Risk\n\n### 3. Rating Behavior\n- **39% of orders receive no rating** — the single largest data gap\n- Predictive modeling shows rating likelihood is weakly predictable from order features\n- Rating absence correlates with lower engagement, not necessarily dissatisfaction\n\n### 4. Operational Consistency\n- Weekday vs weekend differences exist but are **practically small** (small effect sizes despite statistical significance)\n- Food preparation time (not delivery) is the **more variable component** of total wait time\n- Cuisine type drives more variation in prep time than day of week"),
    ("md", "## Methodology\n\n| Technique | Application |\n|---|---|\n| Non-parametric hypothesis testing | Weekday/weekend comparisons, cuisine differences |\n| Chi-square independence tests | Cuisine-day and rating-day associations |\n| Rule-based segmentation (FMS) | Customer behavioral grouping |\n| Binary classification (Logistic, RF, GBM) | Rating prediction with 5-fold stratified CV |\n| 56 automated tests | Data validation, function correctness, model integrity |"),
    ("md", "## Recommendations\n\n1. **Invest in retention over acquisition**: A 10% improvement in one-time customer conversion would add ~78 repeat customers — targeting them with personalized re-engagement based on their first order's cuisine preference\n\n2. **Incentivize ratings**: The 39% gap limits analytical capability. Consider post-delivery prompts timed to when food is consumed, not when it arrives\n\n3. **Diversify restaurant partnerships**: Top-10 concentration at 46% is a risk. Actively promote mid-tier restaurants through featured placements and promotional pricing\n\n4. **Focus operational improvement on prep time consistency**: Delivery time variation is harder to control (depends on distance, traffic). Prep time is within restaurant control and shows the most cuisine-to-cuisine variance"),
    ("md", "## Technical Stack\n\n- **Analysis**: pandas, scipy, scikit-learn\n- **Visualization**: matplotlib, seaborn (20 plot functions)\n- **Dashboard**: Streamlit (4-tab interactive exploration)\n- **Testing**: pytest (56 tests across 7 test files)\n- **Project structure**: Modular `src/` with data, features, analysis, models, and visualization subpackages"),
    ("md", "## Limitations\n\n- Dataset contains **1,898 orders** — sufficient for EDA and statistical testing, but constrains ML model complexity\n- **No temporal granularity**: only weekday/weekend, no timestamps for time-of-day or seasonal analysis\n- **No customer demographics**: segmentation relies solely on transactional behavior\n- **Single snapshot**: no ability to track customer lifecycle or measure campaign impact over time"),
])

# Write all notebooks
for name, nb in [
    ("00_data_overview", nb00),
    ("01_univariate_eda", nb01),
    ("02_bivariate_eda", nb02),
    ("03_statistical_tests", nb03),
    ("04_customer_segments", nb04),
    ("05_predictive_modeling", nb05),
    ("06_final_report", nb06),
]:
    path = NOTEBOOKS_DIR / f"{name}.ipynb"
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"Created {path.name}")
