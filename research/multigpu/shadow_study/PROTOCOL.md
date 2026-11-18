# Shadow allocation pilot — frozen before replay timings, November 4, 2026

Goal: test whether observed traffic or a causal forecast can generate better
maps than the fixed boundary-aware reference. No live ownership switching.
Use the unchanged mode-3 physics/exchange backend, full-N capacity per rank.
Three physical A100-80GB GPUs. Four existing test-seed Berkeley scenarios,
16,384 requested trips, distributed/concentrated/mixed/bursty. These scenarios
were previously inspected: this is a new mechanism pilot, not fresh unseen-data
confirmation. No predictor fitting or hyperparameter tuning on replay outcomes.

Capture fixed-map trajectories every 30 s through t=480 s. Decision times
180, 300 and 420 s; use only current state, preceding 30 s and known departures.
Replay each identical decision snapshot for 60 simulated seconds, dt=.25 s.
Fixed, reactive and forecast maps share kernels, inputs and pool provisioning.
Five shuffled unprofiled repeats per candidate; separate diagnostics. Verify
restart continuity against the uninterrupted fixed run and exact final equality
across every candidate/repeat. Every-step reference checks precede timing.

Reactive: estimate junction work from current owned records (including pending
and completed) and recent rank-local GPU elapsed time per visible record. This
is a rank-average attribution proxy, NOT directly measured per-junction cost.
Cut costs use current road residents, lane reservations and recent road departures.
Forecast: same mechanism, but replace road/connector counts by mean expected
occupancy over 60 s: current count plus a bounded linear trend of non-spawn
net membership change, plus known scheduled departures. Cap by lane storage
and total requested population; subtract consumed pending records. No future
simulation state is read. Forecasting is a simple causal rule, not a learned model.

Both maps use weighted METIS, 5% balance target, seed 2710. Relabel the three
partitions to minimize current vehicle ownership changes. Count changed junctions,
vehicle authority changes and lower-bound vehicle payload (432 bytes/vehicle).
Do not convert payload to a claimed migration time: rebuilding ghosts, indexes,
reservations and synchronization are not implemented as a live switch.

Primary comparison: paired replay complete-step time + each candidate's own
construction/forecast time + conservatively charged observation capture time
+ per-decision share of one-time immutable-topology loading
versus fixed; forecast must also beat reactive. Migration is excluded only for
an optimistic bound, explicitly labeled. Report the remaining millisecond
budget a real switch would have to fit inside, before claiming net gains.
Also show cold replay process wall+construction costs, not as a migration proxy.
At least 5% pre-migration improvement with positive case-bootstrap lower bound
is the gate for investing in a measured migration implementation. Passing this
gate would not prove a deployable dynamic allocator. Windows are clustered by
four demand cases for uncertainty, not treated as 12 independent networks.

No new budget: cumulative posted study spend $10.85496491 / $15.
Reserve at most $3.64: 1000 s function +120 s startup +60 s cleanup at
$0.003/s +$0.10. One container, no retries, deadline and partial artifacts.
