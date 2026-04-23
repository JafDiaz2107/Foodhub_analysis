# Data Dictionary

## Source: `data/raw/foodhub_order.csv`

| Column | Type | Range | Description |
|---|---|---|---|
| `order_id` | int | 1477070–1484843 | Unique order identifier |
| `customer_id` | int | 52832–405881 | Customer identifier (not sequential) |
| `restaurant_name` | str | 178 unique | Restaurant name |
| `cuisine_type` | str | 14 categories | American, Japanese, Italian, Chinese, Mexican, Indian, Mediterranean, Middle Eastern, Thai, French, Southern, Korean, Spanish, Vietnamese |
| `cost_of_the_order` | float | ~4.47–35.41 | Order cost in USD |
| `day_of_the_week` | str | Weekday, Weekend | Binary day classification |
| `rating` | str→float | 3, 4, 5, or "Not given" | Customer satisfaction rating. Converted to float (NaN) at load time |
| `food_preparation_time` | int | 20–35 min | Time from order to food ready |
| `delivery_time` | int | 15–33 min | Time from restaurant to customer |

## Engineered Features (from `src/features/build.py`)

| Feature | Derivation |
|---|---|
| `has_rating` | 1 if rating is not NaN, else 0 |
| `total_time` | food_preparation_time + delivery_time |
| `cost_per_minute` | cost_of_the_order / total_time |
| `prep_delivery_ratio` | food_preparation_time / delivery_time |
| `is_weekend` | 1 if day_of_the_week == "Weekend" |
| `customer_order_count` | Number of orders by this customer |
| `is_repeat_customer` | 1 if customer_order_count > 1 |
| `customer_avg_spend` | Mean cost across customer's orders |

## Customer-Level Metrics (from `src/analysis/segments.py`)

| Metric | Derivation |
|---|---|
| `order_count` | Count of orders per customer |
| `total_spend` | Sum of order costs |
| `avg_spend` | Mean order cost |
| `avg_prep_time` | Mean preparation time |
| `avg_delivery_time` | Mean delivery time |
| `avg_rating` | Mean rating (NaN if never rated) |
| `rating_count` | Number of rated orders |
| `weekend_pct` | % of orders placed on weekends |
| `preferred_cuisine` | Most frequently ordered cuisine |
