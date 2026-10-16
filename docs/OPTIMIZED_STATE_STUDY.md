# Optimized state-cost rerun — 2026-10-16

Simulation checkpoint: `4ab1d4e`; `LPSIM_MOTION_CONTEXT=3`. Frozen inputs and physical references: `artifacts/current/berkeley-state-study-20261014`. New outputs: `artifacts/current/berkeley-optimized-state-20261016`; reports: `reports/berkeley-optimized-state-20261016`. Earlier milestone outputs are never overwritten.

The experiment first replays the original 36 physical windows, including regional/state filters, speed controls, population controls, completed padding, storage permutations and launch sizes. Additional fixtures test approximately 90% stopped road vehicles and pending padding. These are performance fixtures, not calibrated traffic trajectories. Pending padding adds vehicles whose departure is beyond the observation time; it isolates inactive-record costs without changing current road traffic.

The full Berkeley matrix retains distributed 512/2,048 and concentrated 2,048 demand, seeds 1401/1402, controllers 0/2, identical paired routes/departures, 0.25 s steps and 600 s horizon. Repeat 0 supplies trajectory/checkpoint observations, repeats 1–3 timing. Exact physical CSV comparison with the original study is required. Index construction is inside timed GPU intervals.

All-kernel Nsight samples restart one complete step at 60, 180 and 450 s. Unlike the earlier filtered profiles, these capture all sorting, indexing, motion and junction kernels. They use cache-control all for reproducible replay; compare earlier cache-control-none counters only with that protocol difference explicit. SourceCounters are collected separately for two 180 s cases. Six state-filtered motion profiles per case retain full context, including inactive records. These filtered costs are not additive pieces of full-run time.

Motion scan counters use a separately instrumented executable. Receiving-loop visits are expected to be zero; remaining connector-loop visits follow the incoming-lane list. Instrumentation timing is excluded from normal timings. Additional state features include occupied lanes, unique queried movements, conflict-neighbor counts, approaching leaders, queued followers and near-stop vehicles with occupied receiving lanes. State categories overlap; occupancy of a downstream lane is not proof that it is the sole cause of a stop.

Predictors compare requested-only, active-count, topology, observed-state, observed-work and pre-run-prior features. Fixed ridge penalty 0.01; seed 1402 held out, with a second split holding out regions 12–15. Separate exploratory predictors target complete-step and component times using median matched-window event times from three repeats (18 train/18 test observations, no geographic-transfer claim). Pre-run priors retain their original point-queue parameters and see no observed future states.

Allocation remains conditional on useful held-out improvement and an identified workload that can actually be scheduled independently. A regression fit or lower stall percentage alone does not authorize a speedup claim or justify an allocator implementation. Whole-step measured overhead, physical equivalence, synchronization and index maintenance must be included in any later allocation experiment.

Commands:

```sh
python3 tools/optimized_state_study.py prepare
python3 tools/optimized_state_study.py
python3 tools/optimized_extra_controls.py
MPLCONFIGDIR=/tmp/lpsim-mpl python3 tools/report_optimized_state_study.py
```

Fixed-clock wrappers store and restore settings in `finally`; never run timing and profiler stages concurrently. Use a new output ID for independent repetitions. Source hashes, binary hashes, input hashes and environment are retained with the artifact.
