"""
generate_data.py
────────────────
Creates a synthetic retail-sales CSV (retail_sales.csv) spanning one calendar
year (2025-01-01 → 2025-12-31) with realistic patterns:

• 5 product categories across 3 stores
• Weekday/weekend seasonality, holiday spikes, summer lift
• Random inventory dips (stockout events) injected ~8 % of records
• Columns: date, store_id, product_category, units_sold, unit_price,
           inventory_level, is_promotion, day_of_week, month
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── dimensions ───────────────────────────────────────────────────────────────
dates = pd.date_range("2025-01-01", "2025-12-31", freq="D")
stores = ["STORE_A", "STORE_B", "STORE_C"]
categories = {
    "Electronics":   {"base_sales": 30, "price": 149.99},
    "Clothing":      {"base_sales": 55, "price":  39.99},
    "Groceries":     {"base_sales": 120, "price":   8.49},
    "Home & Garden": {"base_sales": 25, "price":  64.99},
    "Toys":          {"base_sales": 40, "price":  24.99},
}

# US-style holidays that drive spikes
holidays = pd.to_datetime([
    "2025-01-01", "2025-02-14", "2025-05-26", "2025-07-04",
    "2025-09-01", "2025-11-27", "2025-11-28", "2025-12-24",
    "2025-12-25", "2025-12-31",
])

rows = []
for date in dates:
    for store in stores:
        for cat, meta in categories.items():
            base = meta["base_sales"]

            # ── seasonal multiplier (summer bump, Dec holiday surge) ──────
            month = date.month
            season = 1.0
            if month in (6, 7, 8):
                season = 1.25            # summer
            elif month == 12:
                season = 1.60            # holiday rush
            elif month in (1, 2):
                season = 0.85            # post-holiday slump

            # ── weekend lift ──────────────────────────────────────────────
            dow = date.dayofweek          # 0=Mon … 6=Sun
            weekend_mult = 1.30 if dow >= 5 else 1.0

            # ── holiday spike ─────────────────────────────────────────────
            holiday_mult = 1.80 if date in holidays else 1.0

            # ── random promotion (~12 %) ──────────────────────────────────
            is_promo = int(np.random.rand() < 0.12)
            promo_mult = 1.35 if is_promo else 1.0

            # ── final units sold ──────────────────────────────────────────
            units = int(
                base * season * weekend_mult * holiday_mult * promo_mult
                * np.random.uniform(0.70, 1.30)
            )
            units = max(units, 0)

            # ── price with small noise ────────────────────────────────────
            price = round(meta["price"] * np.random.uniform(0.95, 1.05), 2)

            # ── inventory level ───────────────────────────────────────────
            # Normal range 80-300; ~8 % chance of a severe dip (stockout zone)
            if np.random.rand() < 0.08:
                inv = int(np.random.uniform(0, 15))   # stockout / near-stockout
            else:
                inv = int(np.random.uniform(80, 300))

            rows.append({
                "date":             date.strftime("%Y-%m-%d"),
                "store_id":         store,
                "product_category": cat,
                "units_sold":       units,
                "unit_price":       price,
                "inventory_level":  inv,
                "is_promotion":     is_promo,
                "day_of_week":      date.day_name(),
                "month":            month,
            })

            # ── sprinkle in ~2 % dirty rows (missing / negative) ─────────
            if np.random.rand() < 0.02:
                rows[-1]["units_sold"] = None        # missing value
            if np.random.rand() < 0.01:
                rows[-1]["unit_price"] = -abs(price)  # negative price

df = pd.DataFrame(rows)

# shuffle lightly so the CSV isn't perfectly sorted
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

out = Path(__file__).resolve().parent / "retail_sales.csv"
df.to_csv(out, index=False)
print(f"✔  Generated {len(df):,} rows → {out}")
print(f"   Columns : {list(df.columns)}")
print(f"   Date range : {df['date'].min()} → {df['date'].max()}")
print(f"   Null cells : {df.isnull().sum().sum()}")
