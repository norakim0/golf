"""Run every analysis step in order (01 -> 05).

How to use in IDLE: File > Open this file, then Run > Run Module (F5).
Step 01 is skipped automatically if the raw Excel file is not in data/raw/.
Figures appear in outputs/figures/, tables and text reports in outputs/reports/.
"""
import sys
import runpy
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(HERE))           # so "from common import ..." works

STEPS = ["01_clean_utilization.py", "02_sql_analysis.py", "03_time_study.py",
         "04_spc_charts.py", "05_queue_sim.py"]
RAW = BASE / "data" / "raw" / "Golfzon_Raw_data.xlsx"

for step in STEPS:
    print("\n" + "#" * 70)
    print("RUNNING", step)
    print("#" * 70)
    if step.startswith("01") and not RAW.exists():
        print("SKIPPED: raw Excel not found at", RAW)
        print("Using the existing data/clean/clean.csv instead.")
        continue
    try:
        runpy.run_path(str(HERE / step), run_name="__main__")
    except Exception:
        print(f"\nERROR in {step}:")
        traceback.print_exc()
        print("Stopping. Fix the error above, then run again.")
        break
else:
    print("\nAll steps finished. Open outputs/figures and outputs/reports.")
