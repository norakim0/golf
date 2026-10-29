import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = str(Path(__file__).resolve().parents[1])
RAW = f"{BASE}/data/raw/Golfzon_Raw_data.xlsx"  # place the raw file here (not tracked by git)
CLEAN = f"{BASE}/data/clean/clean.csv"
FIG = f"{BASE}/outputs/figures"
REP = f"{BASE}/outputs/reports"
for d in (os.path.dirname(CLEAN), FIG, REP):
    os.makedirs(d, exist_ok=True)

# 1. load
raw = pd.read_excel(RAW)
raw.columns = ["dt_raw", "room_raw", "type", "mode", "players", "holes_raw", "rounds"]
n_raw = len(raw)

# 2. time parsing (cross-midnight handled)
def parse_row(s):
    date_s, t = s[:10], s[11:]
    st_s, en_s = t.split("~")
    start = pd.Timestamp(f"{date_s} {st_s}")
    end = pd.Timestamp(f"{date_s} {en_s}")
    crosses = end < start
    if crosses:
        end += pd.Timedelta(days=1)
    return start, end, crosses

parsed = raw["dt_raw"].apply(parse_row)
df = raw.copy()
df["start"] = [p[0] for p in parsed]
df["end"] = [p[1] for p in parsed]
df["crosses_midnight"] = [p[2] for p in parsed]
df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60

# 3. room and holes
room_map = {"(무지개홀)": "rainbow", "1": "room1", "2": "room2", "3": "room3"}
df["room"] = df["room_raw"].astype(str).map(room_map)
cap = {"rainbow": 3, "room1": 4, "room2": 2, "room3": 4}
df["capacity"] = df["room"].map(cap)
df["over_capacity"] = df["players"] > df["capacity"]

hm = df["holes_raw"].str.extract(r"\[(\d+)/(\d+)\]").astype(int)
df["holes_played"] = hm[0]
df["holes_total"] = hm[1]
df["is_full_18"] = df["holes_played"] == 18
df["expected_rounds"] = df["players"] * df["holes_played"] / 18
df["rounds_mismatch"] = (df["rounds"] - df["expected_rounds"]).abs() > 0.5

df["weekday"] = df["start"].dt.day_name()
df["is_weekend"] = df["start"].dt.weekday >= 5
df["start_hour"] = df["start"].dt.hour
df["group"] = np.where(df["players"] == 1, "solo", "group")
hour_f = df["start"].dt.hour + df["start"].dt.minute / 60
df["price_tier"] = pd.cut(hour_f, bins=[0, 13, 16, 24],
                          labels=["before13", "13-16", "after16"], right=False).astype(str)

# 4. minute-level occupancy grid
t0 = df["start"].min().normalize()
t1 = df["end"].max().normalize() + pd.Timedelta(days=1)
N = int((t1 - t0) / pd.Timedelta(minutes=1))
minutes = t0 + pd.to_timedelta(np.arange(N), unit="min")
mod = minutes.hour * 60 + minutes.minute
is_wk = minutes.weekday >= 5
open_min = np.where(is_wk, 8 * 60, 10 * 60 + 30)
close_min = 23 * 60
is_open = (mod >= open_min) & (mod < close_min)

rooms = ["rainbow", "room1", "room2", "room3"]
count = {r: np.zeros(N, dtype=np.int16) for r in rooms}
for _, row in df.iterrows():
    a = int((row["start"] - t0) / pd.Timedelta(minutes=1))
    b = int((row["end"] - t0) / pd.Timedelta(minutes=1))
    count[row["room"]][a:min(b, N)] += 1

occ = {r: (count[r] > 0) for r in rooms}
overlap = {r: int((count[r] > 1).sum()) for r in rooms}

# 5. utilization (denominator = operating minutes)
open_total = int(is_open.sum())
util_rows = []
for r in rooms:
    o = int((occ[r] & is_open).sum())
    util_rows.append({
        "room": r, "capacity": cap[r], "open_minutes": open_total,
        "occupied_minutes_in_open": o,
        "utilization_pct": round(o / open_total * 100, 1),
        "games": int((df["room"] == r).sum()),
        "overlap_minutes": overlap[r],
    })
all_occ = sum(int((occ[r] & is_open).sum()) for r in rooms)
util_rows.append({
    "room": "ALL", "capacity": None, "open_minutes": open_total * 4,
    "occupied_minutes_in_open": all_occ,
    "utilization_pct": round(all_occ / (open_total * 4) * 100, 1),
    "games": len(df), "overlap_minutes": sum(overlap.values()),
})
util = pd.DataFrame(util_rows)

# 6. weekday x hour heatmap
wd_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
occ_any = sum((occ[r] & is_open).astype(int) for r in rooms)
grid = pd.DataFrame({"wd": minutes.weekday, "hour": minutes.hour,
                     "open": is_open, "occ": occ_any})
g = grid[grid["open"]].groupby(["wd", "hour"]).agg(occ=("occ", "sum"), slots=("open", "size"))
g["util_pct"] = g["occ"] / (g["slots"] * 4) * 100
heat = g["util_pct"].unstack("hour").reindex(range(7))
heat.index = wd_names

fig, ax = plt.subplots(figsize=(12, 4.5))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
ax.set_xticks(range(heat.shape[1])); ax.set_xticklabels(heat.columns)
ax.set_yticks(range(7)); ax.set_yticklabels(heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = heat.values[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
ax.set_xlabel("Hour of day"); ax.set_title("Utilization (%) by weekday and hour, all 4 rooms")
fig.colorbar(im, ax=ax, label="%")
plt.tight_layout(); plt.savefig(f"{FIG}/heatmap_weekday_hour.png", dpi=150); plt.close()

# 7. solo vs group
grp = df.groupby("group").agg(games=("players", "size"), rounds=("rounds", "sum"))
grp["games_share_pct"] = (grp["games"] / grp["games"].sum() * 100).round(1)
grp["rounds_share_pct"] = (grp["rounds"] / grp["rounds"].sum() * 100).round(1)
pc = df.groupby("players").agg(games=("players", "size"), rounds=("rounds", "sum"))
pc["games_share_pct"] = (pc["games"] / pc["games"].sum() * 100).round(1)

# 8. save
# English labels for the cleaned output
df_out = pd.DataFrame({
    "start_datetime": df["start"].dt.strftime("%Y-%m-%d %H:%M"),
    "end_datetime": df["end"].dt.strftime("%Y-%m-%d %H:%M"),
    "room_id": df["room"].map({"rainbow": "Rainbow Hall", "room1": "Room 1", "room2": "Room 2", "room3": "Room 3"}),
    "game_type": df["type"].map({"일반경기": "Regular", "GLF": "GLF"}),
    "mode": df["mode"].map({"스트로크": "Stroke", "매치": "Match", "포섬": "Foursome"}),
    "players": df["players"],
    "holes_played": df["holes_played"],
    "holes_total": df["holes_total"],
    "rounds_reported": df["rounds"],
    "expected_rounds": df["expected_rounds"].round(2),
    "duration_min": df["duration_min"].round(1),
    "weekday": df["weekday"],
    "is_weekend": df["is_weekend"],
    "start_hour": df["start_hour"],
    "solo_or_group": df["group"].map({"solo": "Solo", "group": "Group"}),
    "price_tier": df["price_tier"].map({"before13": "Before 13:00", "13-16": "13:00-16:00", "after16": "After 16:00"}),
    "room_capacity": df["capacity"],
    "over_capacity": df["over_capacity"],
    "crosses_midnight": df["crosses_midnight"],
    "is_full_18": df["is_full_18"],
    "rounds_mismatch": df["rounds_mismatch"],
})
df_out.to_csv(CLEAN, index=False, encoding="utf-8-sig")
util.to_csv(f"{REP}/room_utilization.csv", index=False, encoding="utf-8-sig")
heat.round(1).to_csv(f"{REP}/heatmap_weekday_hour.csv", encoding="utf-8-sig")
grp.to_csv(f"{REP}/solo_vs_group.csv", encoding="utf-8-sig")
pc.to_csv(f"{REP}/players_distribution.csv", encoding="utf-8-sig")

log = {
    "raw_rows": n_raw,
    "clean_rows": len(df_out),
    "over_capacity_rows": int(df["over_capacity"].sum()),
    "crosses_midnight_rows": int(df["crosses_midnight"].sum()),
    "rounds_mismatch_rows": int(df["rounds_mismatch"].sum()),
    "non_18_hole_rows": int((~df["is_full_18"]).sum()),
    "overlap_minutes_by_room": str(overlap),
    "period": f"{df['start'].min():%Y-%m-%d} ~ {df['start'].max():%Y-%m-%d}",
    "open_minutes_total_per_room": open_total,
}
pd.Series(log).to_csv(f"{REP}/cleaning_log.csv", header=["value"], encoding="utf-8-sig")

print("== CLEANING ==")
for k, v in log.items():
    print(k, ":", v)
print("\n== UTILIZATION ==")
print(util.to_string(index=False))
print("\n== HEATMAP (utilization %) ==")
print(heat.round(1).to_string())
print("\n== SOLO vs GROUP ==")
print(grp.to_string())
print("\n== PLAYERS ==")
print(pc.to_string())
print("\n== ROUNDS MISMATCH (first 10) ==")
print(df.loc[df["rounds_mismatch"], ["start", "room", "players", "holes_played", "rounds", "expected_rounds"]].head(10).to_string())
