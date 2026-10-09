# Milestone 1: valid paired baseline

The frozen simulation commit is recorded in the experiment manifest. This
experiment uses the existing following backend, with no GPU-algorithm changes.
The only simulator addition records pre-step populations and post-step action
counts on the host, outside the CUDA-event timing intervals.

```
python3 tools/milestone1_baseline.py prepare
cmake --build build -j2
# Commit the source and network before freezing the executable.
python3 tools/milestone1_baseline.py freeze
python3 tools/milestone1_baseline.py run
python3 tools/milestone1_baseline.py profile
python3 tools/validate_milestone1.py
python3 tools/report_milestone1.py
```

The dated artifact directory is intentionally not overwritten by `prepare`.
For a new experiment, use an isolated source workspace with `tools/research.py
new-run`, or choose a new unique artifact/network directory in the harness.
`freeze` must precede all measurements; do not replace the binary after running.
Profiling requires counter access through `sudo -n`; ordinary runs require CUDA.
No hardware-counter runs execute concurrently with the timing repeats.

Each of 20, 400 and 1,600 trips runs under the 72-second controller, mode 2
(always-permitted signal movements with conflict and downstream protection),
and a synthetic capacity control. The capacity control comprises 32 disconnected
copies of a straight two-lane, three-road corridor. It preserves the departure
schedule but deliberately changes geometry and routes; it is not an isolated
signal treatment. All its roads and connectors are under `data/networks/`.
The two San Pablo signal modes preserve identical routes, lanes and departures.

Every case has three parallel and three serial unprofiled runs, with alternate
case ordering between repeats. Parallel precedes serial within each pair.
Checksums of states, events, signals and time series must match across all six.
Full trajectories from the first parallel run are retained; other repetitions
retain their checksums, metrics, logs and every-step population/action observations.

Nsight separately samples six kernel launches at each of 60, 180 and 450 seconds.
The launch offset is `time / 0.25 * 6`. The sampled step's populations and actions
must match the unprofiled reference. One sample per window is descriptive and
does not support counter uncertainty estimates. Profiler CSVs preserve launch
geometry and native units; report duration values are converted to microseconds.

`step_metrics.csv` records populations at the start of the step and actions over
the next 0.25 seconds. Pending includes future departures; `not_departed` separates
them from already-due trips. Road stopped means speed below 0.5 m/s. Connector
vehicles are separate. Requested equals pending + road stopped + road moving +
connector + completed. Denials are repeated decisions, not distinct vehicles.

Timing is measured with CUDA events around road propagation and the junction
pipeline; those intervals include launch gaps. Wall timing also includes audits
and disk output. Power mode is retained at 15 W; clocks use DVFS, with initial
settings recorded. These measurements are not fixed-clock microbenchmarks.

The report presents associations between traffic state and GPU counters.
The source still scans the vehicle array for leaders for every road-active car,
including stopped cars. Speed is therefore not an adequate proxy for work.
Pending and completed records also remain in launched arrays. Milestone 2 is
needed to separate population, movement mix and data-layout effects causally.

To replay retained inputs/binary without regenerating the old demand:

```
python3 tools/replay_milestone1.py --case controller72_20
```

This rewrites the model path for the local archive, creates a fresh output folder,
and requires exact recorded state/event/signal/time-series hashes. Use `--serial`
to run the reference. It does not require the pre-baseline demand-generation tree.

Network dimensions differ: San Pablo has 59 roads, 132 movements and 21 junction
records; the synthetic control has 96 roads, 128 movements and 128 junction records.
Both use 128-thread blocks. In particular, differing arbiter work prevents a
movement-only interpretation of corridor versus San Pablo performance.
