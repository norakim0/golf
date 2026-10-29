SCREEN GOLF VENUE OPERATIONS ANALYSIS
=====================================

An end-to-end industrial engineering analysis of a 4-room screen golf venue
(594 sessions, 2026-07-18 to 2026-10-04). Question: where is capacity wasted,
when does demand peak, and what happens to waiting time if demand grows or a
room goes out of service?

Everything runs with plain Python and IDLE. No paid software and no coding
environment other than IDLE is needed. See docs/IDLE_GUIDE.txt for a click-by-click guide.

MODULES
-------
01 clean_utilization  Parse times, flag data issues (never delete), room utilization,
                      weekday x hour heatmap, solo vs group mix.
02 sql_analysis       SQLite queries: GROUP BY, CASE WHEN, CTEs, window functions (LAG).
03 time_study         Round duration by group size, OLS regression with robust standard
                      errors, distribution fitting, Lean waste metrics.
04 spc_charts         p-chart and I-MR control charts with out-of-control rules.
05 queue_sim          Discrete-event simulation: validation, 200 replications, what-if.

HOW TO RUN (IDLE)
-----------------
1. Put the project folder anywhere (for example Documents/golf_repo).
2. Open IDLE, File > Open, choose src/00_check_setup.py, Run > Run Module (F5).
3. Fix anything marked [MISSING], then open src/run_all.py and press F5.
4. Open outputs/figures (charts) and outputs/reports (tables and text summaries).
The raw Excel file is optional: data/clean/clean.csv is included, and step 01 is
skipped automatically when the raw file is absent.

KEY RESULTS (all produced by the scripts; re-run to verify)
-----------------------------------------------------------
Utilization (module 01)
- Overall utilization 19.0% of operating minutes (Rainbow Hall 22.6%, Room 1 25.7%,
  Room 3 23.1%, Room 2 4.5%).
- Group sessions are 54.9% of games but 73.9% of rounds played.

Time study (module 03, 563 complete stroke-play rounds)
- Each additional player adds about 47 minutes (95% CI 45.3 to 49.2); adjusted R2 = 0.866.
- Room 2 rounds run about 25 minutes longer than Room 1 at the same group size
  (95% CI 20.4 to 30.5, 37 sessions in Room 2). This is an association; the cause
  (equipment problems, player mix) is not identified by the data.
- GLF-tournament rounds are about 5 minutes longer than regular rounds.
- Early exits: 5.1% of sessions (30 of 594), 3.6% of booked minutes.
- Seat fill while a room is in use: 58.8%. Solo sessions in 3-4 player rooms: 39.4% of sessions.
- Gaps between sessions in the same room and day: median 71 min; 21.6% are 10 min or less.

SPC (module 04)
- One day outside the p-chart limits (2026-09-16: 3 of 4 sessions ended early).
- No daily-occupancy points outside I-MR limits (weekday and weekend charts separately).

Simulation (module 05)
- Validation: simulated 7.49 sessions/day vs observed 7.52; utilization 18.6% vs 19.0%.
- Mean wait (minutes) at observed demand / 2x demand:
  current layout 0.81 / 7.31; Room 2 out of service 3.50 / 24.88; one added 4-player room 0.17 / 2.30.
- Multipliers above 1.0 are stress tests, not forecasts.

LIMITATIONS
-----------
- The file records when games started, not arrivals, walk-aways, or reservations,
  so true demand during busy moments may be higher than observed.
- 594 sessions over 79 days is about 7.5 per day; confirm the export is the complete record.
- Closing time of 23:00 is an assumption; 17 sessions end after 23:00.
- The simulation pools service times across rooms, so it does not model Room 2 being slower.
- Cleaning and reset time between games is not recorded.
- The simulation is written in pure Python (not SimPy, Arena, or Simio).

PRIVACY
-------
data/raw/ is excluded by .gitignore. Share this project publicly only with the
venue owner's permission. The cleaned file contains no customer names.

ROADMAP
-------
[x] 01 cleaning and utilization   [x] 02 SQL   [x] 03 time study   [x] 04 SPC   [x] 05 simulation
[ ] 06 demand forecasting (baseline vs model, error metric)
[ ] 07 slot and price optimization (linear programming)
[ ] 08 customer segmentation (clustering)
[ ] Room 2 repair-period comparison (needs repair dates)
