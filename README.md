# FoodHub Order Analysis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Food delivery platform analytics: EDA, statistical hypothesis testing, customer segmentation, and predictive modeling on 1,898 orders across 178 restaurants and 14 cuisines.

## Key Findings

1. **Demand concentration** — Top 10 restaurants handle 46% of all orders. American and Japanese cuisines account for 45%+ of volume.
2. **Retention crisis** — 65% of customers ordered only once. Repeat customers spend more on average.
3. **Rating gap** — 39% of orders receive no rating. Predictive modeling shows rating likelihood is weakly predictable from order features alone.
4. **Operational consistency** — Weekday/weekend differences are statistically significant but practically small. Prep time (not delivery) is the more variable component.

## Methodology

| Technique | Application |
|---|---|
| Non-parametric hypothesis testing (Mann-Whitney U, Kruskal-Wallis) | Weekday/weekend and cross-cuisine comparisons |
| Chi-square independence tests | Cuisine-day and rating-day associations |
| Rule-based FMS segmentation | Customer behavioral grouping (Frequency-Monetary-Satisfaction) |
| Binary classification (Logistic Regression, Random Forest, GBM) | Rating prediction with 5-fold stratified CV |

## Project Structure

```
foodhub-analysis/
├── src/
│   ├── data/load.py                # Load CSV, validate schema, convert rating
│   ├── features/build.py           # Clean, engineer features, encode
│   ├── analysis/
│   │   ├── stats.py                # Hypothesis tests, correlations, effect sizes
│   │   └── segments.py             # Rule-based FMS customer segmentation
│   ├── models/predict.py           # Rating classification, cost regression
│   └── visualization/plots.py      # 20 reusable plot functions
├── notebooks/
│   ├── 00_data_overview.ipynb
│   ├── 01_univariate_eda.ipynb
│   ├── 02_bivariate_eda.ipynb
│   ├── 03_statistical_tests.ipynb
│   ├── 04_customer_segments.ipynb
│   ├── 05_predictive_modeling.ipynb
│   └── 06_final_report.ipynb
├── dashboard/app.py                # 4-tab Streamlit interactive dashboard
├── tests/                          # 56 tests across 7 files
├── data/raw/foodhub_order.csv      # 1,898 orders × 9 columns
└── docs/
    ├── ARCHITECTURE.md
    └── DATA_DICTIONARY.md
```

## Dataset

**FoodHub Order Dataset** — 1,898 orders from an online food delivery aggregator.

| Column | Type | Description |
|---|---|---|
| `order_id` | int | Unique order identifier |
| `customer_id` | int | Customer identifier |
| `restaurant_name` | str | Restaurant name (178 unique) |
| `cuisine_type` | str | Cuisine category (14 types) |
| `cost_of_the_order` | float | Order cost in USD |
| `day_of_the_week` | str | "Weekday" or "Weekend" |
| `rating` | float | Customer rating (3-5, or NaN if not given) |
| `food_preparation_time` | int | Prep time in minutes |
| `delivery_time` | int | Delivery time in minutes |

## Setup

```bash
git clone https://github.com/JafDiaz2107/Foodhub_analysis.git
cd Foodhub_analysis

# Install
make install

# Run tests
make test

# Launch notebooks
make notebook

# Run dashboard
make dashboard
```

## Design Decisions

1. **Non-parametric tests** — Data is small (1,898 rows) with skewed distributions. Mann-Whitney U, Kruskal-Wallis, and Spearman correlations are honest choices over t-tests and Pearson.
2. **Rule-based segmentation over clustering** — With 65% one-time customers and only 9 features, clustering produces unstable results. FMS tiers are interpretable and directly actionable by marketing teams.
3. **Binary rating prediction** — Predicting "rated vs. not rated" (61/39 split) instead of actual rating values (only 1,150 valid ratings clustered at 3-5). Honest framing over impressive-sounding but poorly-performing ordinal classification.
4. **Stratified 5-fold CV** — Single train/test split on 1,898 rows is too noisy. Cross-validation provides reliable estimates with confidence intervals.

## Tech Stack

- **Python 3.11+** — Primary language
- **pandas / numpy** — Data manipulation
- **scipy** — Statistical testing
- **scikit-learn** — Classification, model evaluation
- **matplotlib / seaborn** — Visualization
- **streamlit** — Interactive dashboard
- **pytest** — Testing (56 tests)
- **ruff** — Linting

## License

MIT
