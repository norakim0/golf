"""Module 3 - Time study: regression, distribution fitting, and Lean waste analysis.

Industrial engineering skills shown:
  * Time study / work measurement (observed cycle times, planning percentiles)
  * Multiple linear regression built from matrix algebra (NumPy), with classical
    and heteroskedasticity-robust (HC1) standard errors, 95% CIs, R-squared
  * Distribution fitting (normal / lognormal / gamma) with AIC and KS tests (SciPy)
  * Lean waste metrics: early exits (defect-like), under-filled seats (unused
    capacity), and changeover gaps between games (waiting / idle time)

Run in IDLE: open this file and press F5. Needs data/clean/clean.csv.
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from common import load_clean, save_table, header, ensure_dirs, FIG, REP

ensure_dirs()
df = load_clean()
lines = []


def say(text=""):
    """Print and also keep the line for the text report."""
    print(text)
    lines.append(text)


# ----------------------------------------------------------------------------
# 0. Analysis sample
# ----------------------------------------------------------------------------
# NOTE: use df["mode"], not df.mode - df.mode is a pandas method!
stroke = df[(df["mode"] == "Stroke") & (df["duration_min"] > 0)].copy()
complete = stroke[stroke["is_complete"]].copy()

header("0. ANALYSIS SAMPLE")
say(f"All sessions in file                     : {len(df)}")
say(f"Stroke-play sessions with duration > 0   : {len(stroke)}")
say(f"Complete rounds (>= 17 of 18 holes)      : {len(complete)}  <- used for time study")
say("Rule: 17/18 holes counted as complete (last hole likely unrecorded).")


# ----------------------------------------------------------------------------
# 1. Planning times by group size (time-study table)
# ----------------------------------------------------------------------------
header("1. ROUND DURATION BY GROUP SIZE (minutes)")
complete["min_per_player"] = complete["duration_min"] / complete["players"]
by_players = (complete.groupby("players")
              .agg(sessions=("duration_min", "size"), mean=("duration_min", "mean"),
                   median=("duration_min", "median"), std=("duration_min", "std"),
                   p90=("duration_min", lambda s: s.quantile(0.90)),
                   max=("duration_min", "max"),
                   minutes_per_player=("min_per_player", "mean")).round(1))
say(by_players.to_string())
save_table(by_players.reset_index(), "t1_duration_by_group_size.csv")


# ----------------------------------------------------------------------------
# 2. Multiple linear regression (OLS from scratch)
# ----------------------------------------------------------------------------
def ols(y, X, names):
    """OLS with classical and HC1-robust standard errors."""
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    resid = y - X @ beta
    dof = n - k
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(xtx_inv) * s2)
    meat = (X * resid[:, None] ** 2).T @ X                  # sum e_i^2 x_i x_i'
    se_rob = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv) * n / dof)   # HC1
    t = beta / se_rob
    p = 2 * stats.t.sf(np.abs(t), dof)
    tcrit = stats.t.ppf(0.975, dof)
    ss_res, ss_tot = resid @ resid, ((y - y.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    table = pd.DataFrame({
        "term": names, "coef": beta, "robust_se": se_rob, "t": t, "p_value": p,
        "ci95_low": beta - tcrit * se_rob, "ci95_high": beta + tcrit * se_rob})
    stats_ = {"n": n, "k": k, "r2": r2, "adj_r2": 1 - (1 - r2) * (n - 1) / dof,
              "rmse_min": float(np.sqrt(s2))}
    return table, stats_, resid


def design(d, full):
    cols = {"intercept": np.ones(len(d)), "players": d["players"].astype(float).to_numpy()}
    if full:
        cols["GLF_game"] = (d["game_type"] == "GLF").astype(float).to_numpy()
        cols["tier_13_16"] = (d["price_tier"] == "13:00-16:00").astype(float).to_numpy()
        cols["tier_after_16"] = (d["price_tier"] == "After 16:00").astype(float).to_numpy()
        cols["weekend"] = d["is_weekend_day"].astype(float).to_numpy()
        cols["room_rainbow"] = (d["room_id"] == "Rainbow Hall").astype(float).to_numpy()
        cols["room_2"] = (d["room_id"] == "Room 2").astype(float).to_numpy()
        cols["room_3"] = (d["room_id"] == "Room 3").astype(float).to_numpy()
    names = list(cols)
    return np.column_stack([cols[c] for c in names]), names


header("2. REGRESSION: duration_min ~ drivers")
y = complete["duration_min"].to_numpy(dtype=float)

XA, na = design(complete, full=False)
tab_a, st_a, _ = ols(y, XA, na)
XB, nb = design(complete, full=True)
tab_b, st_b, resid_b = ols(y, XB, nb)

say("Model A (players only)")
say(tab_a.round(3).to_string(index=False))
say(f"n={st_a['n']}  R2={st_a['r2']:.3f}  adj R2={st_a['adj_r2']:.3f}  RMSE={st_a['rmse_min']:.1f} min")
say("")
say("Model B (players + game type + price tier + weekend + room)")
say("Reference levels: Regular game, before 13:00, weekday, Room 1")
say(tab_b.round(3).to_string(index=False))
say(f"n={st_b['n']}  R2={st_b['r2']:.3f}  adj R2={st_b['adj_r2']:.3f}  RMSE={st_b['rmse_min']:.1f} min")
save_table(tab_a, "t2a_regression_model_A.csv")
save_table(tab_b, "t2b_regression_model_B.csv")

# Sensitivity: drop rows flagged as over capacity
ok = complete[~complete["over_capacity"]]
Xs, ns = design(ok, full=True)
tab_s, st_s, _ = ols(ok["duration_min"].to_numpy(dtype=float), Xs, ns)
p_all = tab_b.loc[tab_b.term == "players", "coef"].iloc[0]
p_ok = tab_s.loc[tab_s.term == "players", "coef"].iloc[0]
say("")
say(f"Sensitivity: players coefficient = {p_all:.2f} min/player (all rows) vs "
    f"{p_ok:.2f} (excluding {len(complete) - len(ok)} over-capacity rows).")

# Which non-group-size terms are statistically distinguishable from zero (5% level)?
sig = tab_b[(tab_b.term != "intercept") & (tab_b.term != "players") & (tab_b.p_value < 0.05)]
say("Non-group-size terms with p < 0.05: " + (", ".join(sig.term) if len(sig) else "none"))
say("Caution: coefficients describe association in this venue's data, not proven causes.")


# ----------------------------------------------------------------------------
# 3. Distribution fitting of minutes per player
# ----------------------------------------------------------------------------
header("3. DISTRIBUTION FIT: minutes per player (complete rounds)")
pp = (complete["duration_min"] / complete["players"]).to_numpy(dtype=float)
fits = []
for name, dist, fix in [("normal", stats.norm, False),
                        ("lognormal", stats.lognorm, True),
                        ("gamma", stats.gamma, True)]:
    params = dist.fit(pp, floc=0) if fix else dist.fit(pp)
    loglik = dist.logpdf(pp, *params).sum()
    k = len(params) - (1 if fix else 0)
    ks = stats.kstest(pp, dist.cdf, args=params)
    fits.append({"distribution": name, "params": tuple(round(float(p), 3) for p in params),
                 "aic": round(2 * k - 2 * loglik, 1), "ks_stat": round(ks.statistic, 3),
                 "ks_p": round(ks.pvalue, 3)})
fit_df = pd.DataFrame(fits).sort_values("aic")
say(fit_df.to_string(index=False))
best = fit_df.iloc[0]["distribution"]
say(f"Lowest AIC: {best}. (KS p-values are optimistic because parameters were fitted on the same data.)")
save_table(fit_df, "t3_distribution_fit.csv")


# ----------------------------------------------------------------------------
# 4. Lean waste analysis
# ----------------------------------------------------------------------------
header("4. LEAN WASTE ANALYSIS")

# 4a. Early exits (defect-like: round not completed)
early = df[df["is_early_exit"]]
say("4a. Early exits (fewer than 17 holes played)")
say(f"    {len(early)} of {len(df)} sessions ({len(early) / len(df):.1%}); "
    f"mean duration {early['duration_min'].mean():.1f} min; "
    f"share of all booked minutes {early['duration_min'].sum() / df['duration_min'].sum():.1%}")

# 4b. Under-filled seats (unused capacity)
d = df.copy()
d["seats_used"] = np.minimum(d["players"], d["room_capacity"])
d["seat_min"] = d["seats_used"] * d["duration_min"]
d["cap_min"] = d["room_capacity"] * d["duration_min"]
sums = d.groupby("room_id")[["seat_min", "cap_min"]].sum()
seat_room = (sums["seat_min"] / sums["cap_min"]).round(3)
seat_all = d["seat_min"].sum() / d["cap_min"].sum()
say("4b. Seat fill rate while a room is in use (players / capacity, time-weighted)")
say(f"    Overall {seat_all:.1%}.  By room: " + ", ".join(f"{k} {v:.1%}" for k, v in seat_room.items()))
solo_big = d[(d["players"] == 1) & (d["room_capacity"] >= 3)]
say(f"    Solo sessions in rooms that hold 3-4 players: {len(solo_big)} "
    f"({len(solo_big) / len(d):.1%} of sessions)")
save_table(seat_room.reset_index(name="seat_fill_rate"), "t4b_seat_fill_by_room.csv")

# 4c. Gaps between consecutive sessions in the same room on the same day
g = df.sort_values(["room_id", "start_datetime"]).copy()
g["prev_end"] = g.groupby("room_id")["end_datetime"].shift()
g["gap_min"] = (g["start_datetime"] - g["prev_end"]).dt.total_seconds() / 60
gaps = g[g["prev_end"].dt.normalize() == g["start_datetime"].dt.normalize()]["gap_min"].dropna()
say("4c. Gap between sessions (same room, same day)")
say(f"    {len(gaps)} gaps; median {gaps.median():.0f} min; "
    f"P25 {gaps.quantile(.25):.0f}; P75 {gaps.quantile(.75):.0f}; max {gaps.max():.0f}")
say(f"    Back-to-back (<= 10 min): {(gaps <= 10).sum()} ({(gaps <= 10).mean():.1%}); "
    f"long idle (> 120 min): {(gaps > 120).sum()} ({(gaps > 120).mean():.1%})")
say("    Cleaning/reset time is not recorded, so the 10-minute cut-off is an assumption.")
save_table(gaps.describe().round(1).reset_index(), "t4c_gap_summary.csv")


# ----------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.5))
jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(complete))
ax.scatter(complete["players"] + jitter, complete["duration_min"], s=12, alpha=0.4)
xs = np.array([1, 5])
ax.plot(xs, tab_a.coef[0] + tab_a.coef[1] * xs, color="crimson",
        label=f"Fit: {tab_a.coef[0]:.1f} + {tab_a.coef[1]:.1f} x players")
ax.set_xlabel("Players in session"); ax.set_ylabel("Round duration (min)")
ax.set_title("18-hole duration vs group size (complete rounds)")
ax.legend(); plt.tight_layout(); plt.savefig(FIG / "time_study_regression.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(pp, bins=30, density=True, alpha=0.5, label="Observed")
grid = np.linspace(pp.min(), pp.max(), 300)
for name, dist, fix in [("normal", stats.norm, False), ("lognormal", stats.lognorm, True),
                        ("gamma", stats.gamma, True)]:
    prm = dist.fit(pp, floc=0) if fix else dist.fit(pp)
    ax.plot(grid, dist.pdf(grid, *prm), label=name)
ax.set_xlabel("Minutes per player"); ax.set_ylabel("Density")
ax.set_title("Distribution fit: minutes per player")
ax.legend(); plt.tight_layout(); plt.savefig(FIG / "time_study_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(gaps.clip(upper=360), bins=36)
ax.set_xlabel("Gap before next session (min, capped at 360)"); ax.set_ylabel("Count")
ax.set_title("Gaps between sessions, same room and day")
plt.tight_layout(); plt.savefig(FIG / "time_study_gaps.png", dpi=150); plt.close()

(REP / "summary_03_time_study.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"\n[saved] {REP / 'summary_03_time_study.txt'}")
print("[saved] figures: time_study_regression.png, time_study_distribution.png, time_study_gaps.png")
