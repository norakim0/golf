"""Module 4 - Statistical Process Control (SPC) charts.

Industrial engineering / quality skills shown:
  * p-chart with variable subgroup size (daily early-exit rate)
  * Individuals and Moving Range (I-MR) chart (daily occupied minutes),
    stratified by weekday vs weekend because the two have different means
  * Out-of-control rules: Rule 1 (point beyond 3-sigma limits) and
    Rule 2 (8 consecutive points on one side of the center line)

Run in IDLE: open this file and press F5. Needs data/clean/clean.csv.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import load_clean, save_table, header, ensure_dirs, all_days, FIG, REP

ensure_dirs()
df = load_clean()
lines = []


def say(text=""):
    print(text)
    lines.append(text)


# ----------------------------------------------------------------------------
# 1. Daily summary table
# ----------------------------------------------------------------------------
daily = (df.groupby("date")
         .agg(sessions=("players", "size"),
              early_exits=("is_early_exit", "sum"),
              occupied_min=("duration_min", "sum"))
         .reindex(all_days(df), fill_value=0))
daily.index.name = "date"
daily["weekend"] = daily.index.weekday >= 5
daily["early_exits"] = daily["early_exits"].astype(int)

header("1. DAILY SUMMARY")
say(f"Calendar days: {len(daily)} ({int((~daily.weekend).sum())} weekdays, {int(daily.weekend.sum())} weekend days)")
say(f"Days with zero sessions: {int((daily.sessions == 0).sum())}")
say(f"Mean sessions/day: weekday {daily.loc[~daily.weekend, 'sessions'].mean():.1f}, "
    f"weekend {daily.loc[daily.weekend, 'sessions'].mean():.1f}")
save_table(daily.reset_index(), "t5_daily_summary.csv")


# ----------------------------------------------------------------------------
# 2. p-chart: proportion of sessions that end early
# ----------------------------------------------------------------------------
header("2. p-CHART: daily early-exit rate")
pdat = daily[daily.sessions > 0].copy()
p_bar = pdat.early_exits.sum() / pdat.sessions.sum()
pdat["p"] = pdat.early_exits / pdat.sessions
sigma = np.sqrt(p_bar * (1 - p_bar) / pdat.sessions)
pdat["ucl"] = p_bar + 3 * sigma
pdat["lcl"] = np.maximum(0, p_bar - 3 * sigma)
pdat["out"] = (pdat.p > pdat.ucl) | (pdat.p < pdat.lcl)

say(f"Center line p-bar = {p_bar:.3f} ({pdat.early_exits.sum()} early exits / {pdat.sessions.sum()} sessions)")
say(f"Days outside 3-sigma limits: {int(pdat.out.sum())}")
for d, r in pdat[pdat.out].iterrows():
    say(f"   {d:%Y-%m-%d} ({d:%a}): {int(r.early_exits)} of {int(r.sessions)} sessions ended early "
        f"({r.p:.0%}); UCL {r.ucl:.0%}")

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.plot(pdat.index, pdat.p, marker="o", ms=3, lw=0.8, label="Daily rate")
ax.step(pdat.index, pdat.ucl, where="mid", color="crimson", lw=1, label="UCL (3 sigma)")
ax.axhline(p_bar, color="green", lw=1, label="Center (p-bar)")
ax.scatter(pdat.index[pdat.out], pdat.p[pdat.out], color="crimson", zorder=5, label="Out of control")
ax.set_ylabel("Early-exit rate"); ax.set_title("p-chart: share of sessions ending before 17 holes")
ax.legend(loc="upper right", fontsize=8); plt.tight_layout()
plt.savefig(FIG / "spc_p_chart.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------
# 3. I-MR chart on daily occupied minutes, by day type
# ----------------------------------------------------------------------------
def imr(series):
    """Return center, UCL, LCL for an individuals chart (d2 = 1.128 -> factor 2.66)."""
    mr = series.diff().abs().dropna()
    center, mr_bar = series.mean(), mr.mean()
    return center, center + 2.66 * mr_bar, max(0.0, center - 2.66 * mr_bar), mr, mr_bar


def run_rule(series, center, length=8):
    """True where the last `length` points (incl. this one) are all on one side."""
    side = np.sign(series.to_numpy() - center)
    flags = np.zeros(len(series), dtype=bool)
    for i in range(length - 1, len(series)):
        w = side[i - length + 1: i + 1]
        if (w > 0).all() or (w < 0).all():
            flags[i] = True
    return flags


header("3. I-MR CHART: daily occupied minutes (all rooms), by day type")
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
flag_rows = []
for ax, (label, wk) in zip(axes, [("Weekdays", False), ("Weekend days", True)]):
    s = daily.loc[daily.weekend == wk, "occupied_min"].astype(float)
    center, ucl, lcl, mr, mr_bar = imr(s)
    rule1 = (s > ucl) | (s < lcl)
    rule2 = pd.Series(run_rule(s, center), index=s.index)
    say(f"{label}: n={len(s)}, center {center:.0f} min, UCL {ucl:.0f}, LCL {lcl:.0f}, "
        f"mean moving range {mr_bar:.0f}")
    say(f"   Rule 1 (beyond limits): {int(rule1.sum())} days; "
        f"Rule 2 (8 in a row on one side): {int(rule2.sum())} days")
    for d in s.index[rule1]:
        say(f"   Rule 1 -> {d:%Y-%m-%d} ({d:%a}): {s[d]:.0f} min")
        flag_rows.append({"date": d.date(), "day_type": label, "rule": "Rule 1", "occupied_min": s[d]})
    for d in s.index[rule2 & ~rule1]:
        flag_rows.append({"date": d.date(), "day_type": label, "rule": "Rule 2", "occupied_min": s[d]})
    ax.plot(s.index, s.values, marker="o", ms=3, lw=0.8)
    ax.axhline(center, color="green", lw=1); ax.axhline(ucl, color="crimson", lw=1)
    ax.axhline(lcl, color="crimson", lw=1, ls="--")
    ax.scatter(s.index[rule1], s[rule1], color="crimson", zorder=5)
    ax.set_title(f"I chart: daily occupied minutes - {label}")
    ax.set_ylabel("Minutes")
plt.tight_layout(); plt.savefig(FIG / "spc_imr_chart.png", dpi=150); plt.close()
save_table(pd.DataFrame(flag_rows), "t5_spc_flags.csv")

say("")
say("Reading the charts: a flagged day is a signal to ask 'what was different that day?'")
say("(holiday, tournament, equipment fault, weather). The data alone cannot name the cause.")
say("Note: if Rule 2 flags appear, they usually indicate a gradual shift (e.g., seasonal drift), not a one-day special cause.")

(REP / "summary_04_spc.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] {REP / 'summary_04_spc.txt'}")
print("[saved] figures: spc_p_chart.png, spc_imr_chart.png")
