import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_pipeline_in_isolated_directory(tmp_path):
    for name in ("generate_data.py", "etl_pipeline.py", "forecast_model.py", "visualize.py", "run_pipeline.py"):
        shutil.copy2(ROOT / name, tmp_path / name)
    subprocess.run([sys.executable, str(tmp_path / "run_pipeline.py")], cwd=tmp_path, check=True, timeout=300)
    with sqlite3.connect(tmp_path / "sales.db") as conn:
        count, low, high = conn.execute("SELECT COUNT(*), MIN(stockout_probability), MAX(stockout_probability) FROM sales_predictions").fetchone()
        assert count > 0
        assert 0 <= low <= high <= 1
    for name in ("monthly_revenue_trend.png", "stockout_heatmap.png", "forecast_risk_dashboard.png"):
        assert (tmp_path / "output" / name).stat().st_size > 0
