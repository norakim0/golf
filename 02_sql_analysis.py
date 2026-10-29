"""Module 2A - SQL analysis (SQLite, built into Python; no install needed).

IE/analyst skills shown: SELECT/GROUP BY, CASE WHEN, CTEs, window functions
(LAG, RANK, moving average), date arithmetic, joining logic into reusable queries.
"""
import sqlite3
from common import load_clean, save_table, header, ensure_dirs
import pandas as pd

ensure_dirs()
df = load_clean()

# SQLite stores dates as text, so format them explicitly.
sql_df = df[["start_datetime", "end_datetime", "room_id", "game_type", "players",
             "holes_played", "duration_min", "price_tier", "is_weekend_day"]].copy()
sql_df["start_datetime"] = sql_df["start_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
sql_df["end_datetime"] = sql_df["end_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
sql_df["is_weekend_day"] = sql_df["is_weekend_day"].astype(int)

con = sqlite3.connect(":memory:")
sql_df.to_sql("sessions", con, index=False)
header(f"SQLite version {sqlite3.sqlite_version} (window functions need >= 3.25)")

queries = {}

# Q1: GROUP BY with conditional aggregation (CASE WHEN)
queries["q1_tier_summary"] = """
SELECT price_tier,
       COUNT(*)                                             AS sessions,
       ROUND(AVG(duration_min), 1)                          AS avg_duration_min,
       SUM(CASE WHEN players = 1 THEN 1 ELSE 0 END)         AS solo_sessions,
       SUM(CASE WHEN players >= 2 THEN 1 ELSE 0 END)        AS group_sessions,
       ROUND(100.0 * SUM(CASE WHEN players = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS solo_pct
FROM sessions
GROUP BY price_tier
ORDER BY sessions DESC;
"""

# Q2: window function LAG = gap (changeover/idle) between consecutive games in the same room
queries["q2_room_gaps"] = """
WITH ordered AS (
    SELECT room_id, start_datetime, end_datetime,
           LAG(end_datetime) OVER (PARTITION BY room_id ORDER BY start_datetime) AS prev_end
    FROM sessions
)
SELECT room_id,
       COUNT(prev_end)                                                         AS gaps,
       ROUND(AVG((julianday(start_datetime) - julianday(prev_end)) * 1440), 1) AS avg_gap_min,
       SUM(CASE WHEN (julianday(start_datetime) - julianday(prev_end)) * 1440 <= 10 THEN 1 ELSE 0 END) AS back_to_back_le_10min
FROM ordered
WHERE prev_end IS NOT NULL
  AND date(prev_end) = date(start_datetime)          -- same-day gaps only (ignore overnight)
GROUP BY room_id
ORDER BY room_id;
"""

# Q3: CTE + 7-day moving average (window frame)
queries["q3_daily_moving_avg"] = """
WITH daily AS (
    SELECT date(start_datetime) AS day, COUNT(*) AS sessions, SUM(players) AS players
    FROM sessions
    GROUP BY date(start_datetime)
)
SELECT day, sessions, players,
       ROUND(AVG(sessions) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW), 2) AS sessions_7d_avg
FROM daily
ORDER BY day;
"""

# Q4: RANK() window to find the busiest start hours within weekday/weekend
queries["q4_peak_hours"] = """
WITH h AS (
    SELECT is_weekend_day AS weekend,
           CAST(strftime('%H', start_datetime) AS INTEGER) AS hour,
           COUNT(*) AS sessions
    FROM sessions
    GROUP BY is_weekend_day, CAST(strftime('%H', start_datetime) AS INTEGER)
)
SELECT weekend, hour, sessions,
       RANK() OVER (PARTITION BY weekend ORDER BY sessions DESC) AS rank_in_daytype
FROM h
ORDER BY weekend, rank_in_daytype
LIMIT 12;
"""

for name, q in queries.items():
    result = pd.read_sql_query(q, con)
    header(name)
    print(result.head(15).to_string(index=False))
    save_table(result, f"{name}.csv")

con.close()
print("\nDone. Open outputs/reports/ to see each query result as a CSV.")
