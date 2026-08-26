"""
visualize.py
────────────
Generates three publication-quality KPI charts from the SQLite database
and saves them as PNGs in the output/ folder:

  1. monthly_revenue_trend.png   – Monthly revenue by product category (stacked area)
  2. stockout_heatmap.png        – Stockout frequency heatmap (store × category × month)
  3. forecast_risk_dashboard.png – Stockout-risk distribution + feature importance
"""

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "sales.db"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

# ── global style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#181b25",
    "axes.edgecolor":   "#2d3040",
    "axes.labelcolor":  "#c8cad0",
    "text.color":       "#e0e2e8",
    "xtick.color":      "#8b8fa0",
    "ytick.color":      "#8b8fa0",
    "grid.color":       "#252838",
    "grid.alpha":       0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
})

# Curated palette (vibrant yet harmonious)
PALETTE = ["#6366f1", "#22d3ee", "#f472b6", "#facc15", "#34d399"]

conn = sqlite3.connect(DB_PATH)

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1 — Monthly Revenue Trend (Stacked Area)
# ═══════════════════════════════════════════════════════════════════════════
print("── Chart 1: Monthly Revenue Trend ──────────────────────")
df_rev = pd.read_sql_query("""
    SELECT strftime('%Y-%m', date) AS ym,
           product_category,
           SUM(revenue) AS total_revenue
    FROM sales
    GROUP BY ym, product_category
    ORDER BY ym
""", conn)

pivot = df_rev.pivot(index="ym", columns="product_category", values="total_revenue").fillna(0)

fig, ax = plt.subplots(figsize=(14, 6))
ax.stackplot(
    range(len(pivot)),
    *[pivot[c].values for c in pivot.columns],
    labels=pivot.columns,
    colors=PALETTE,
    alpha=0.85,
)
ax.set_xticks(range(len(pivot)))
ax.set_xticklabels(pivot.index, rotation=45, ha="right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
ax.set_title("Monthly Revenue by Product Category", fontsize=16, fontweight="bold", pad=16)
ax.set_xlabel("Month")
ax.set_ylabel("Revenue")
ax.legend(loc="upper left", framealpha=0.3, fontsize=9)
ax.grid(axis="y", linestyle="--")
fig.tight_layout()
fig.savefig(OUT_DIR / "monthly_revenue_trend.png", dpi=180)
plt.close(fig)
print("   ✔ monthly_revenue_trend.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2 — Stockout Heatmap (Store × Category, monthly)
# ═══════════════════════════════════════════════════════════════════════════
print("── Chart 2: Stockout Heatmap ───────────────────────────")
df_so = pd.read_sql_query("""
    SELECT store_id,
           product_category,
           month,
           SUM(stockout_flag) AS stockout_events
    FROM sales
    GROUP BY store_id, product_category, month
""", conn)

# Build a combined label for rows: "STORE_A | Electronics"
df_so["label"] = df_so["store_id"] + " │ " + df_so["product_category"]
heat = df_so.pivot(index="label", columns="month", values="stockout_events").fillna(0)
heat.columns = [f"M{int(m):02d}" for m in heat.columns]

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(heat.values, aspect="auto", cmap="magma_r", interpolation="nearest")

ax.set_xticks(range(heat.shape[1]))
ax.set_xticklabels(heat.columns, fontsize=9)
ax.set_yticks(range(heat.shape[0]))
ax.set_yticklabels(heat.index, fontsize=8)

# Annotate cells
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = int(heat.values[i, j])
        color = "white" if val > heat.values.max() * 0.6 else "#c8cad0"
        ax.text(j, i, str(val), ha="center", va="center", fontsize=7, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
cbar.set_label("Stockout Events", color="#c8cad0")
cbar.ax.yaxis.set_tick_params(color="#8b8fa0")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b8fa0")

ax.set_title("Stockout Frequency  (Store × Category × Month)", fontsize=15, fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "stockout_heatmap.png", dpi=180)
plt.close(fig)
print("   ✔ stockout_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3 — Forecast Risk Dashboard (dual panel)
# ═══════════════════════════════════════════════════════════════════════════
print("── Chart 3: Forecast Risk Dashboard ────────────────────")

# 3a. Stockout probability distribution
df_pred = pd.read_sql_query("""
    SELECT stockout_probability, product_category
    FROM sales_predictions
""", conn)

# 3b. Feature importance
df_imp = pd.read_csv(OUT_DIR / "feature_importance.csv").head(12)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5),
                                gridspec_kw={"width_ratios": [1.1, 1]})

# ── Left: probability distribution per category ──────────────────────────
cats = df_pred["product_category"].unique()
for idx, cat in enumerate(sorted(cats)):
    subset = df_pred.loc[df_pred["product_category"] == cat, "stockout_probability"]
    ax1.hist(subset, bins=40, alpha=0.55, color=PALETTE[idx % len(PALETTE)],
             label=cat, edgecolor="none")
ax1.axvline(0.5, color="#ef4444", ls="--", lw=1.5, label="Decision threshold")
ax1.set_title("Predicted Stockout Probability Distribution", fontsize=13, fontweight="bold")
ax1.set_xlabel("Stockout Probability")
ax1.set_ylabel("Frequency")
ax1.legend(fontsize=8, framealpha=0.3)
ax1.grid(axis="y", linestyle="--")

# ── Right: horizontal bar – feature importance ───────────────────────────
colors = [PALETTE[i % len(PALETTE)] for i in range(len(df_imp))]
ax2.barh(df_imp["feature"][::-1], df_imp["importance"][::-1], color=colors[::-1],
         edgecolor="none", height=0.65)
ax2.set_title("Top Feature Importances", fontsize=13, fontweight="bold")
ax2.set_xlabel("Importance")
ax2.grid(axis="x", linestyle="--")

fig.suptitle("Stockout Risk Forecast — Dashboard", fontsize=17, fontweight="bold",
             color="#e0e2e8", y=1.01)
fig.tight_layout()
fig.savefig(OUT_DIR / "forecast_risk_dashboard.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("   ✔ forecast_risk_dashboard.png")

conn.close()
print("\n✔  All 3 charts saved to output/")
