"""
etl_pipeline.py
───────────────
Extract → Transform → Load pipeline:

1. Reads  retail_sales.csv
2. Cleans nulls, negative prices, type coercion
3. Engineers features (revenue, rolling averages, stockout flag)
4. Loads clean data into SQLite (sales.db)
5. Runs aggregation queries and exports summary tables as CSVs
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV_IN = ROOT / "retail_sales.csv"
DB_PATH = ROOT / "sales.db"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. EXTRACT
# ═══════════════════════════════════════════════════════════════════════════
print("── EXTRACT ─────────────────────────────────────────────")
df = pd.read_csv(CSV_IN, parse_dates=["date"])
print(f"   Rows loaded        : {len(df):,}")
print(f"   Null cells (total) : {df.isnull().sum().sum()}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRANSFORM
# ═══════════════════════════════════════════════════════════════════════════
print("\n── TRANSFORM ───────────────────────────────────────────")

# 2a. Drop rows where units_sold is missing (unrecoverable)
before = len(df)
df.dropna(subset=["units_sold"], inplace=True)
print(f"   Dropped {before - len(df)} rows with null units_sold")

# 2b. Fix negative prices → absolute value
neg_mask = df["unit_price"] < 0
print(f"   Fixed {neg_mask.sum()} negative-price rows")
df.loc[neg_mask, "unit_price"] = df.loc[neg_mask, "unit_price"].abs()

# 2c. Cast types
df["units_sold"] = df["units_sold"].astype(int)
df["inventory_level"] = df["inventory_level"].astype(int)
df["is_promotion"] = df["is_promotion"].astype(int)
df["month"] = df["month"].astype(int)

# 2d. Feature engineering
df["revenue"] = (df["units_sold"] * df["unit_price"]).round(2)

# Stockout flag: inventory < 20 units
df["stockout_flag"] = (df["inventory_level"] < 20).astype(int)

# 7-day rolling average sales per store×category
df.sort_values(["store_id", "product_category", "date"], inplace=True)
df["rolling_avg_7d"] = (
    df.groupby(["store_id", "product_category"])["units_sold"]
      .transform(lambda s: s.rolling(7, min_periods=1).mean())
      .round(1)
)

# Days since last stockout (per store×category)
df["days_since_stockout"] = (
    df.groupby(["store_id", "product_category"])["stockout_flag"]
      .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount())
)

print(f"   Rows after cleaning : {len(df):,}")
print(f"   New columns added   : revenue, stockout_flag, rolling_avg_7d, days_since_stockout")

# ═══════════════════════════════════════════════════════════════════════════
# 3. LOAD → SQLite
# ═══════════════════════════════════════════════════════════════════════════
print("\n── LOAD → SQLite ───────────────────────────────────────")
conn = sqlite3.connect(DB_PATH)
df.to_sql("sales", conn, if_exists="replace", index=False)
print(f"   Written to {DB_PATH}  (table: sales)")

# ═══════════════════════════════════════════════════════════════════════════
# 4. SQL AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n── SQL AGGREGATIONS ────────────────────────────────────")

queries = {
    "monthly_revenue": """
        SELECT
            strftime('%Y-%m', date) AS year_month,
            store_id,
            product_category,
            SUM(revenue)           AS total_revenue,
            SUM(units_sold)        AS total_units,
            AVG(inventory_level)   AS avg_inventory,
            SUM(stockout_flag)     AS stockout_days
        FROM sales
        GROUP BY year_month, store_id, product_category
        ORDER BY year_month, store_id, product_category
    """,
    "category_summary": """
        SELECT
            product_category,
            SUM(revenue)                          AS total_revenue,
            ROUND(AVG(units_sold), 1)             AS avg_daily_units,
            SUM(stockout_flag)                    AS total_stockout_events,
            ROUND(100.0 * SUM(stockout_flag) / COUNT(*), 2) AS stockout_pct
        FROM sales
        GROUP BY product_category
        ORDER BY total_revenue DESC
    """,
    "store_performance": """
        SELECT
            store_id,
            SUM(revenue)                          AS total_revenue,
            SUM(units_sold)                       AS total_units,
            COUNT(DISTINCT date)                  AS active_days,
            SUM(stockout_flag)                    AS total_stockout_events
        FROM sales
        GROUP BY store_id
        ORDER BY total_revenue DESC
    """,
}

for name, sql in queries.items():
    result = pd.read_sql_query(sql, conn)
    out_path = OUT_DIR / f"{name}.csv"
    result.to_csv(out_path, index=False)
    print(f"   {name:25s} → {len(result):>5} rows  → {out_path.name}")

conn.close()
print("\n✔  ETL pipeline complete.")
