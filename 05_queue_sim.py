"""Module 5 - Queueing / capacity simulation (discrete-event, Monte Carlo).

Industrial engineering skills shown:
  * Input modeling: time-varying Poisson arrival rates by day type and hour,
    empirical (bootstrap) sampling of group size and service time
  * Discrete-event simulation of multi-server, capacity-constrained rooms
  * Replications, 95% confidence intervals, and model VALIDATION against observed data
  * What-if analysis: demand growth x room configuration (a repair, or a new room)

Pure Python + NumPy (no SimPy needed). Run in IDLE with F5. Needs data/clean/clean.csv.

IMPORTANT LIMITATION: the data records when games STARTED, not when customers ARRIVED
or walked away. At ~19% utilization waits are rare, so the baseline model assumes
observed starts equal arrivals. Real demand during busy moments may be higher than observed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (load_clean, save_table, header, ensure_dirs, all_days,
                    OPEN_WEEKDAY_MIN, OPEN_WEEKEND_MIN, CLOSE_MIN, FIG, REP)

ensure_dirs()
N_REPS = 200
SEED = 2026
MULTIPLIERS = [1.0, 1.5, 2.0, 3.0]
CONFIGS = {                       # room capacities (max players per room)
    "Current (3,4,2,4)": [3, 4, 2, 4],
    "Room 2 out of service (3,4,4)": [3, 4, 4],
    "Add one 4-player room (3,4,2,4,4)": [3, 4, 2, 4, 4],
}
lines = []


def say(text=""):
    print(text)
    lines.append(text)


df = load_clean()
days = all_days(df)
n_days = len(days)
day0 = days[0]
is_wk = np.array([d.weekday() >= 5 for d in days])
open_min = np.where(is_wk, OPEN_WEEKEND_MIN, OPEN_WEEKDAY_MIN)
open_total = float((CLOSE_MIN - open_min).sum())            # open minutes per room, whole period

# ----------------------------------------------------------------------------
# 1. Input modeling
# ----------------------------------------------------------------------------
header("1. INPUT MODELING")
starts = df.assign(hour=df["start_datetime"].dt.hour, wk=df["is_weekend_day"])
counts = starts.groupby(["wk", "hour"]).size()
days_of_type = {False: int((~is_wk).sum()), True: int(is_wk.sum())}
lam = np.zeros((2, 24))                                      # sessions per hour, by day type
for (wk, hour), c in counts.items():
    lam[int(wk), int(hour)] = c / days_of_type[bool(wk)]
say(f"Period: {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d} ({n_days} days)")
say(f"Expected arrivals/day: weekday {lam[0].sum():.1f}, weekend {lam[1].sum():.1f}")
say("Arrival rate = observed starts per hour / number of days of that type (Poisson).")

pool_players = df["players"].to_numpy()
pool_dur = df["duration_min"].to_numpy(dtype=float)
say(f"Service-time pool: {len(pool_dur)} observed sessions; each draw keeps the (players, duration) pair.")
lam_cells = np.array([lam[int(w), h] for w in is_wk for h in range(24)])   # n_days*24 cells


def busy_in_open(start, end):
    """Minutes of [start, end) that fall inside the opening window of the start day."""
    day = int(start // 1440)
    if day >= n_days:
        return 0.0
    lo = day * 1440 + open_min[day]
    hi = day * 1440 + CLOSE_MIN
    return max(0.0, min(end, hi) - max(start, lo))


def gen_arrivals(rng, mult):
    cnt = rng.poisson(lam_cells * mult)
    cell = np.repeat(np.arange(len(cnt)), cnt)
    t = (cell // 24) * 1440 + (cell % 24) * 60 + rng.random(len(cell)) * 60
    t.sort()
    idx = rng.integers(0, len(pool_dur), len(t))
    return t, pool_players[idx], pool_dur[idx]


def run_once(caps, t, players, dur):
    """Each group goes to the feasible room that can start it earliest (ties: smallest room)."""
    free = [0.0] * len(caps)
    waits = np.empty(len(t))
    busy = 0.0
    for j in range(len(t)):
        need = min(int(players[j]), max(caps))
        best, best_start = -1, 1e18
        for i, c in enumerate(caps):
            if c >= need:
                st = free[i] if free[i] > t[j] else t[j]
                if st < best_start or (st == best_start and c < caps[best]):
                    best, best_start = i, st
        waits[j] = best_start - t[j]
        free[best] = best_start + dur[j]
        busy += busy_in_open(best_start, best_start + dur[j])
    return waits, busy / (open_total * len(caps))


# ----------------------------------------------------------------------------
# 2. Validation against observed data
# ----------------------------------------------------------------------------
header("2. VALIDATION (baseline model vs observed)")
obs_busy = 0.0
for s, e in zip(df["start_datetime"], df["end_datetime"]):
    obs_busy += busy_in_open((s - day0).total_seconds() / 60, (e - day0).total_seconds() / 60)
obs_util = obs_busy / (open_total * 4)
obs_per_day = len(df) / n_days

rng = np.random.default_rng(SEED)
val_n, val_u = [], []
for _ in range(N_REPS):
    t, p, d = gen_arrivals(rng, 1.0)
    w, u = run_once(CONFIGS["Current (3,4,2,4)"], t, p, d)
    val_n.append(len(t) / n_days)
    val_u.append(u)


def ci(x):
    x = np.asarray(x, dtype=float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


mn, hn = ci(val_n)
mu, hu = ci(val_u)
say(f"Sessions per day : observed {obs_per_day:.2f} | simulated {mn:.2f} +/- {hn:.2f}")
say(f"Utilization      : observed {obs_util:.1%} | simulated {mu:.1%} +/- {hu:.1%}")
ok_n = abs(mn - obs_per_day) <= max(hn, 0.05 * obs_per_day)
ok_u = abs(mu - obs_util) <= max(hu, 0.05 * obs_util)
say("Validation check (within 5% or 95% CI): " + ("PASSED" if ok_n and ok_u else "REVIEW NEEDED"))
say("Utilization here uses the same clipped-to-opening-hours rule for observed and simulated data,")
say("so it can differ slightly from the Module 1 figure.")

# ----------------------------------------------------------------------------
# 3. What-if scenarios
# ----------------------------------------------------------------------------
header(f"3. WHAT-IF SCENARIOS ({N_REPS} replications each)")
rows = []
for m in MULTIPLIERS:
    rng = np.random.default_rng(SEED + int(m * 100))
    store = {k: {"mean": [], "p_wait": [], "p_gt10": [], "p90": [], "util": []} for k in CONFIGS}
    for _ in range(N_REPS):
        t, p, d = gen_arrivals(rng, m)                       # same arrivals for every config (CRN)
        for name, caps in CONFIGS.items():
            w, u = run_once(caps, t, p, d)
            s = store[name]
            s["mean"].append(w.mean()); s["p_wait"].append((w > 0.5).mean())
            s["p_gt10"].append((w > 10).mean()); s["p90"].append(np.percentile(w, 90))
            s["util"].append(u)
    for name in CONFIGS:
        s = store[name]
        row = {"demand_multiplier": m, "configuration": name}
        for key in ["mean", "p_wait", "p_gt10", "p90", "util"]:
            mean_, half = ci(s[key])
            row[key] = mean_
            row[key + "_ci95"] = half
        rows.append(row)

res = pd.DataFrame(rows)
res_out = pd.DataFrame({
    "demand_x": res.demand_multiplier,
    "configuration": res.configuration,
    "mean_wait_min": res["mean"].round(2),
    "mean_wait_ci95": res["mean_ci95"].round(2),
    "pct_wait_gt_0_5min": (100 * res.p_wait).round(1),
    "pct_wait_gt_10min": (100 * res.p_gt10).round(1),
    "p90_wait_min": res.p90.round(1),
    "utilization_pct": (100 * res.util).round(1),
})
say(res_out.to_string(index=False))
save_table(res_out, "t6_queue_scenarios.csv")

# ----------------------------------------------------------------------------
# 4. Figure
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for name in CONFIGS:
    r = res[res.configuration == name]
    axes[0].errorbar(r.demand_multiplier, r["mean"], yerr=r["mean_ci95"], marker="o", capsize=3, label=name)
    axes[1].plot(r.demand_multiplier, 100 * r.p_gt10, marker="o", label=name)
axes[0].set_xlabel("Demand multiplier (1.0 = observed)"); axes[0].set_ylabel("Mean wait (min)")
axes[0].set_title("Mean wait vs demand")
axes[1].set_xlabel("Demand multiplier (1.0 = observed)"); axes[1].set_ylabel("% of groups waiting > 10 min")
axes[1].set_title("Long waits vs demand"); axes[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIG / "queue_sim_scenarios.png", dpi=150); plt.close()

say("")
say("How to read this: at 1.0x the venue rarely makes anyone wait. Higher multipliers are")
say("hypothetical stress tests, NOT forecasts. Use them to see which configuration degrades")
say("more slowly when demand grows. Walk-away customers and reservations are not modeled.")

(REP / "summary_05_queue_sim.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] {REP / 'summary_05_queue_sim.txt'}")
print("[saved] figure: queue_sim_scenarios.png")
