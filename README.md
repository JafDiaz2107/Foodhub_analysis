# FoodHub Order Analysis

Exploratory data analysis of an online food delivery platform to understand demand patterns, restaurant performance, and customer behavior.

## Problem Statement

FoodHub aggregates orders from multiple restaurants. This analysis investigates:
- Which cuisines and restaurants drive the most demand?
- How do delivery times and order costs vary across cuisines?
- What ordering patterns emerge on weekdays vs. weekends?
- Where are the operational bottlenecks in food preparation and delivery?

## Dataset

Order-level records from the FoodHub platform, including cuisine type, order cost, food preparation time, delivery time, customer rating, and day of the week.

## Approach

1. **Data cleaning** — handled missing values, validated data types, identified outliers in preparation and delivery times
2. **Univariate analysis** — distributions of cost, ratings, prep time, and delivery time
3. **Bivariate analysis** — cuisine vs. cost, restaurant vs. delivery time, weekday vs. weekend patterns
4. **Statistical observations** — identified high-demand restaurants, cost outliers, and delivery efficiency gaps

## Key Findings

- A small number of restaurants account for a disproportionate share of total orders
- Weekend orders show different cuisine preferences and higher average costs
- Food preparation time is the primary bottleneck — more variable than delivery time
- Customer ratings cluster high (4-5), with low ratings correlating with longer total wait times

## Tech Stack

- **Python** — pandas, numpy
- **Visualization** — matplotlib, seaborn
- **Environment** — Jupyter Notebook

## Project Structure

```
Foodhub_analysis/
├── README.md
└── Foodhub/
    ├── Foodhub.ipynb    # Full analysis notebook
    └── data/            # Source dataset
```

## License

MIT
