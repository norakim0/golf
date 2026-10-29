"""Shared helpers for all analysis scripts (paths, data loading, venue constants).

Run any script from IDLE with F5. Figures are saved as PNG files (Agg backend),
so nothing depends on a GUI window.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")          # must be set before pyplot is imported anywhere
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
CLEAN = BASE / "data" / "clean" / "clean.csv"
FIG = BASE / "outputs" / "figures"
REP = BASE / "outputs" / "reports"

# Venue facts supplied by the owner
CAPACITY = {"Rainbow Hall": 3, "Room 1": 4, "Room 2": 2, "Room 3": 4}
ROOMS = list(CAPACITY)
OPEN_WEEKDAY_MIN = 10 * 60 + 30     # 10:30
OPEN_WEEKEND_MIN = 8 * 60           # 08:00
CLOSE_MIN = 23 * 60                 # 23:00 (assumption, to be confirmed)


def ensure_dirs():
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)


def load_clean():
    """Load the cleaned session table and add helper columns."""
    if not CLEAN.exists():
        raise FileNotFoundError(
            f"Cannot find {CLEAN}. Run 01_clean_utilization.py first "
            "(raw Excel must be in data/raw/) or copy clean.csv into data/clean/."
        )
    df = pd.read_csv(CLEAN, encoding="utf-8-sig",
                     parse_dates=["start_datetime", "end_datetime"])
    df["date"] = df["start_datetime"].dt.normalize()
    df["is_weekend_day"] = df["start_datetime"].dt.weekday >= 5
    df["is_complete"] = df["holes_played"] >= 17      # 17/18 treated as complete (last hole likely unrecorded)
    df["is_early_exit"] = df["holes_played"] < 17
    return df


def open_window(date, weekend=None):
    """Return (open_minute, close_minute) of the day for a given date."""
    d = pd.Timestamp(date)
    wk = (d.weekday() >= 5) if weekend is None else weekend
    return (OPEN_WEEKEND_MIN if wk else OPEN_WEEKDAY_MIN), CLOSE_MIN


def all_days(df):
    """Every calendar day between the first and last session (inclusive)."""
    return pd.date_range(df["date"].min(), df["date"].max(), freq="D")


def save_table(df, name, index=False):
    ensure_dirs()
    path = REP / name
    df.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"[saved] {path}")


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
