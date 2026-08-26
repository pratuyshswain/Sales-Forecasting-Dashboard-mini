"""
forecast_model.py
─────────────────
Trains a lightweight scikit-learn model to **predict stockout risk**
(binary: will inventory drop below 20 units in the next 7 days?)

Pipeline:
1. Load cleaned data from SQLite
2. Engineer lag / rolling features per store×category
3. Train a GradientBoostingClassifier (CPU-optimised, ~200 trees)
4. Evaluate (classification report, ROC-AUC)
5. Persist model + predictions back to SQLite for the viz layer
"""

import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "sales.db"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ═══════════════════════════════════════════════════════════════════════════
print("── LOADING DATA ────────────────────────────────────────")
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("SELECT * FROM sales ORDER BY store_id, product_category, date", conn)
df["date"] = pd.to_datetime(df["date"])
print(f"   Rows : {len(df):,}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\n── FEATURE ENGINEERING ─────────────────────────────────")

# Target: will there be a stockout (inventory < 20) within the NEXT 7 days?
df["future_stockout_7d"] = (
    df.groupby(["store_id", "product_category"])["stockout_flag"]
      .transform(lambda s: s.rolling(7, min_periods=1).max().shift(-7))
)
df["future_stockout_7d"] = df["future_stockout_7d"].fillna(0).astype(int)

# Lag features
for lag in [1, 3, 7, 14]:
    df[f"sales_lag_{lag}"] = (
        df.groupby(["store_id", "product_category"])["units_sold"]
          .shift(lag)
    )
    df[f"inv_lag_{lag}"] = (
        df.groupby(["store_id", "product_category"])["inventory_level"]
          .shift(lag)
    )

# Rolling statistics
for window in [7, 14, 30]:
    df[f"sales_roll_mean_{window}"] = (
        df.groupby(["store_id", "product_category"])["units_sold"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )
    df[f"sales_roll_std_{window}"] = (
        df.groupby(["store_id", "product_category"])["units_sold"]
          .transform(lambda s: s.rolling(window, min_periods=1).std().fillna(0))
    )
    df[f"inv_roll_mean_{window}"] = (
        df.groupby(["store_id", "product_category"])["inventory_level"]
          .transform(lambda s: s.rolling(window, min_periods=1).mean())
    )

# Encode categoricals
df["store_enc"] = df["store_id"].astype("category").cat.codes
df["cat_enc"] = df["product_category"].astype("category").cat.codes
df["dow_enc"] = df["date"].dt.dayofweek

# ── Feature list ──────────────────────────────────────────────────────────
feature_cols = [
    "units_sold", "unit_price", "inventory_level", "is_promotion",
    "month", "rolling_avg_7d", "days_since_stockout",
    "store_enc", "cat_enc", "dow_enc",
] + [c for c in df.columns if c.startswith(("sales_lag", "inv_lag",
                                             "sales_roll", "inv_roll"))]

# Drop rows with NaN features (from lags at start of series)
df_model = df.dropna(subset=feature_cols + ["future_stockout_7d"]).copy()
print(f"   Rows for modelling : {len(df_model):,}")
print(f"   Features           : {len(feature_cols)}")
print(f"   Target balance     : {df_model['future_stockout_7d'].value_counts().to_dict()}")

X = df_model[feature_cols].values
y = df_model["future_stockout_7d"].values

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════
print("\n── TRAINING ────────────────────────────────────────────")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── EVALUATION ──────────────────────────────────────────")
print(classification_report(y_test, y_pred, target_names=["No Stockout", "Stockout"]))
auc = roc_auc_score(y_test, y_prob)
print(f"   ROC-AUC : {auc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. FEATURE IMPORTANCE → CSV
# ═══════════════════════════════════════════════════════════════════════════
importance = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False)
importance.to_csv(OUT_DIR / "feature_importance.csv", index=False)
print(f"\n   Top-5 features:")
for _, row in importance.head(5).iterrows():
    print(f"      {row['feature']:30s}  {row['importance']:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. PERSIST PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════
df_model["predicted_stockout"] = model.predict(
    df_model[feature_cols].values
)
df_model["stockout_probability"] = model.predict_proba(
    df_model[feature_cols].values
)[:, 1].round(4)

conn = sqlite3.connect(DB_PATH)
df_model.to_sql("sales_predictions", conn, if_exists="replace", index=False)
conn.close()

print(f"\n✔  Model predictions saved to sales.db  (table: sales_predictions)")
print(f"✔  Feature importance  → output/feature_importance.csv")
