# Commands
- Install: `make install`
- Test: `pytest tests/ -v`
- Lint: `ruff check src/ tests/`
- Format: `ruff format src/ tests/`
- Notebook: `jupyter notebook notebooks/`
- Dashboard: `streamlit run dashboard/app.py`

# Code Style
- Type hints on all public function signatures
- Google-style docstrings
- Ruff for linting/formatting (config in pyproject.toml)
- Imports: stdlib → third-party → local, separated by blank lines
- Plot functions return `Figure` objects, never call `plt.show()`

# Structure
- `src/data/` — loading CSV, schema validation
- `src/features/` — cleaning, feature engineering, encoding
- `src/analysis/` — hypothesis tests, correlations, customer segmentation
- `src/models/` — classification (rated vs not), regression (order cost)
- `src/visualization/` — all plotting functions
- `notebooks/` — numbered 00-06, import from src/
- `dashboard/` — Streamlit app
- `tests/` — pytest, mirrors src/ structure

# Gotchas
- Dataset: `data/raw/foodhub_order.csv`, comma delimiter
- `rating` column contains "Not given" strings (39%) — converted to NaN at load time
- Only 1,898 rows — use cross-validation, not single train/test split
- 65% of customers have only 1 order — segmentation must handle this
- Set `random_state` everywhere for reproducibility
