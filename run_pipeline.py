"""
run_pipeline.py
───────────────
Master script – runs the full pipeline in sequence:

    1. generate_data.py   → retail_sales.csv
    2. etl_pipeline.py    → sales.db + summary CSVs
    3. forecast_model.py  → predictions in SQLite
    4. visualize.py       → 3 PNG charts

Usage:
    python run_pipeline.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("1/4  Generating synthetic data", "generate_data.py"),
    ("2/4  Running ETL pipeline",      "etl_pipeline.py"),
    ("3/4  Training forecast model",   "forecast_model.py"),
    ("4/4  Generating visualizations", "visualize.py"),
]


def run(label: str, script: str) -> None:
    path = ROOT / script
    print(f"\n{'═' * 60}")
    print(f"  ▶  {label}")
    print(f"{'═' * 60}")
    result = subprocess.run(
        [sys.executable, str(path)],
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print(f"\n✘  {script} failed (exit code {result.returncode})")
        sys.exit(result.returncode)


if __name__ == "__main__":
    for label, script in STEPS:
        run(label, script)

    print(f"\n{'═' * 60}")
    print("  ✔  PIPELINE COMPLETE")
    print(f"{'═' * 60}")
    print(f"""
  Outputs:
    ├── retail_sales.csv                (raw data)
    ├── sales.db                        (SQLite database)
    └── output/
        ├── monthly_revenue.csv
        ├── category_summary.csv
        ├── store_performance.csv
        ├── feature_importance.csv
        ├── monthly_revenue_trend.png
        ├── stockout_heatmap.png
        └── forecast_risk_dashboard.png
""")
