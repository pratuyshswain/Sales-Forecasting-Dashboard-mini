# Sales Forecasting Dashboard — Mini Prototype

A **local, end-to-end sales forecasting pipeline** that generates synthetic retail data, cleans and aggregates it with pandas + SQLite, trains a stockout-prediction model with scikit-learn, and produces three publication-quality KPI charts.

## What Problem It Solves

In retail, running out of stock (a "stockout") means lost revenue and unhappy customers, but keeping too much stock is expensive. This project anticipates stockouts before they happen by analyzing historical sales trends, seasonality, and inventory levels to predict *if a specific product at a specific store will run out of stock in the next 7 days*. This allows store managers to reorder products proactively.

---

## Architecture

```
generate_data.py          ← 1. Synthetic CSV (1 yr × 3 stores × 5 categories)
       │
       ▼
etl_pipeline.py           ← 2. Clean, transform, load into SQLite
       │
       ▼
forecast_model.py         ← 3. Train GradientBoosting stockout predictor
       │
       ▼
visualize.py              ← 4. Render 3 KPI charts as PNGs
```

All four scripts are chained by **`run_pipeline.py`** — a single command runs the entire pipeline.

---

## Quick Start

### Prerequisites

| Package        | Version  |
|----------------|----------|
| Python         | ≥ 3.9    |
| pandas         | ≥ 1.5    |
| numpy          | ≥ 1.23   |
| scikit-learn   | ≥ 1.2    |
| matplotlib     | ≥ 3.6    |

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

Or run each step individually:

```bash
python generate_data.py      # → retail_sales.csv
python etl_pipeline.py       # → sales.db + output/*.csv
python forecast_model.py     # → predictions in sales.db
python visualize.py          # → output/*.png
```

---

## Output Files

After a successful run you will find:

```
project/
├── retail_sales.csv                  # Raw synthetic data (~5,475 rows)
├── sales.db                          # SQLite database (2 tables)
│
├── output/
│   ├── monthly_revenue.csv           # Aggregated monthly revenue
│   ├── category_summary.csv          # Category-level KPIs
│   ├── store_performance.csv         # Store-level KPIs
│   ├── feature_importance.csv        # Model feature importances
│   │
│   ├── monthly_revenue_trend.png     # Chart 1 – Stacked area
│   ├── stockout_heatmap.png          # Chart 2 – Heatmap
│   └── forecast_risk_dashboard.png   # Chart 3 – Dual panel dashboard
│
├── generate_data.py
├── etl_pipeline.py
├── forecast_model.py
├── visualize.py
├── run_pipeline.py
└── README.md
```

---

## Pipeline Details

### 1. Data Generation (`generate_data.py`)

Generates **one calendar year** (Jan 1 – Dec 31, 2025) of daily retail sales for **3 stores × 5 product categories**:

- **Seasonal effects**: summer lift (+25 %), December surge (+60 %), post-holiday dip (−15 %)
- **Weekend boost**: +30 % on Sat/Sun
- **Holiday spikes**: 10 US holidays (+80 %)
- **Promotions**: ~12 % of rows flagged as promotions (+35 % units)
- **Inventory dips**: ~8 % of rows have near-zero inventory (stockout simulation)
- **Dirty data**: ~2 % null `units_sold`, ~1 % negative prices — for ETL to clean

### 2. ETL Pipeline (`etl_pipeline.py`)

| Step       | Action                                                       |
|------------|--------------------------------------------------------------|
| Extract    | Read `retail_sales.csv` with date parsing                    |
| Transform  | Drop null sales rows, fix negative prices, cast types        |
|            | Engineer: `revenue`, `stockout_flag`, `rolling_avg_7d`, `days_since_stockout` |
| Load       | Write cleaned data to `sales.db → sales` table               |
| Aggregate  | 3 SQL queries produce summary CSVs (monthly, category, store) |

### 3. Forecasting Model (`forecast_model.py`)

- **Target**: Binary — will a stockout occur within the next 7 days?
- **Features** (29 total): lag sales/inventory (1, 3, 7, 14 days), rolling means & std (7, 14, 30 days), promotions, seasonality, encoded store/category
- **Model**: `GradientBoostingClassifier` (200 trees, depth 5, lr 0.1, subsample 0.8)
- **Evaluation**: Precision / Recall / F1 + ROC-AUC
- **Output**: Predictions + probabilities written to `sales_predictions` table in SQLite

### 4. Visualizations (`visualize.py`)

| Chart | File | Description |
|-------|------|-------------|
| 1 | `monthly_revenue_trend.png` | Stacked area chart of monthly revenue by product category |
| 2 | `stockout_heatmap.png` | Heatmap showing stockout events across store × category × month |
| 3 | `forecast_risk_dashboard.png` | Dual panel: stockout probability distribution + top feature importances |

All charts use a dark theme with a curated color palette for high readability and visual appeal.

---

## SQLite Schema

```sql
-- Cleaned sales data
CREATE TABLE sales (
    date TEXT, store_id TEXT, product_category TEXT,
    units_sold INTEGER, unit_price REAL, inventory_level INTEGER,
    is_promotion INTEGER, day_of_week TEXT, month INTEGER,
    revenue REAL, stockout_flag INTEGER,
    rolling_avg_7d REAL, days_since_stockout INTEGER
);

-- Model predictions
CREATE TABLE sales_predictions (
    ...sales columns...,
    predicted_stockout INTEGER,
    stockout_probability REAL
);
```

---

## Tech Stack

| Layer          | Technology                        |
|----------------|-----------------------------------|
| Data gen       | Python, NumPy, pandas             |
| ETL            | pandas, SQLite                    |
| Modelling      | scikit-learn (GradientBoosting)   |
| Visualization  | matplotlib                        |
| Storage        | SQLite (local, zero-config)       |

---

## License

This is a prototype / demo project. Use freely.
