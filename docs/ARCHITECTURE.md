# Architecture

## Pipeline

```
data/raw/foodhub_order.csv
  → src/data/load.py         Load, validate schema, convert rating
  → src/features/build.py    Clean, engineer features, encode
  → src/analysis/stats.py    Hypothesis tests, correlations
  → src/analysis/segments.py Customer segmentation (FMS tiers)
  → src/models/predict.py    Classification, evaluation
  → src/visualization/plots.py  All plotting (returns Figure objects)
```

## Module Responsibilities

| Module | Responsibility | Depends On |
|---|---|---|
| `src/data/load` | Load CSV, validate schema, type conversion | — |
| `src/features/build` | Cleaning, feature engineering, encoding | `data.load` |
| `src/analysis/stats` | Statistical hypothesis tests | `features.build` |
| `src/analysis/segments` | Customer-level aggregation and segmentation | `features.build` |
| `src/models/predict` | ML classification and evaluation | `features.build` |
| `src/visualization/plots` | All plotting functions | — |
| `dashboard/app` | Streamlit interactive app | All `src` modules |
| `notebooks/` | Analysis narrative | All `src` modules |

## Design Principles

1. **Notebooks call `src/`, never the reverse** — Logic lives in modules, narrative lives in notebooks
2. **Plot functions return `Figure` objects** — Callers decide how to display (notebook, dashboard, or save to file)
3. **Statistical functions return dicts** — Standardized keys (statistic, p_value, effect_size, interpretation) for consistent downstream use
4. **Tests mirror `src/` structure** — Each module has a corresponding test file
