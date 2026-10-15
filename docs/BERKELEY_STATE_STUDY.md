# Full Berkeley state-cost study (2026-10-14)

This opt-in research backend lives in `src/simulator/scalable_junction_simulator.cu`.
The existing production backend and frozen Milestones 1 and 2A remain unchanged.
The new backend replaces fixed parallel-junction buffers with network-sized
allocations, uses CSR conflict lists, and stores newly granted movements in
disjoint per-junction spans. Lane keys support eight lanes; the cached Berkeley
input includes five-lane roads. The small serial reference retains its explicit
capacity limit and is used for equivalence tests, not citywide execution.

`tools/test_scalable_junctions.py` checks all 11 original scenarios against the
frozen Milestone 1 executable, scalable serial, scalable parallel, and scalable
parallel with leader indexing. Physical CSVs must be byte-identical, including
the 1,600-trip case, blocked exit, right-on-red, yield, and platoon tests.

The canonical derived model is `data/networks/berkeley_junction_study/`. It is a
research model, not a survey/calibration of Berkeley intersections. Read its
`provenance.json`: short links are contracted, geometry uses endpoint chords,
turn permissions and lane connectors are inferred, and every junction uses the
synthetic controller. OSM edge lengths and lane counts are retained subject to
documented junction trimming. Routes use existing OD pairs, enforce lane
continuity, and are restricted to the backend's 32-road route capacity. Origins
and destinations are citywide OD nodes, not only external network boundaries.

The matrix contains distributed 512-trip, distributed 2,048-trip, and concentrated
2,048-trip demand; seeds 1401/1402; 72 s and always-permitted controllers. Paired
controllers share exactly the same routes and departures. The 600 s horizon and
0.25 s step are fixed. Repeat 0 supplies checkpoints and pilot observations;
repeats 1–3 supply timing statistics. Index construction is charged every step.
All movement/gap checks run throughout; physical output repeats must match.

`tools/berkeley_replay.cu` loads exact ABI-specific checkpoints at 60, 180 and
450 s. It retains all vehicles as read context while filtering road-motion work
by a fixed spatial region or vehicle-state class. Each replay has three warmups
and ten measured identical steps. These regional/state-filtered launches alter
occupancy and cache behavior, so their latencies cannot be added to recover the
original full kernel's time. Separate source instrumentation records receiving
and connector scan counts and per-thread clock deltas; none of its timing is
used for normal performance comparisons. Instrumented and normal physical
digests must match.

Controlled replays additionally change road speed (0 or 5 m/s), inactive padding,
vehicle storage order, active-road population, and block size. Permutations,
padding and block-size variants must preserve canonical physical results.
Population removal is a constructed performance test: it changes neighbor gaps
and can slightly alter the stopped/moving mix. These are not calibrated traffic
trajectories. Motion permutations do not test junction admission priorities.

Nsight samples are separate from timing runs. Full-pipeline samples restart the
exact physical state for one step, so their cache history differs from uninterrupted
simulation. State-filtered motion profiles use warmed immutable replays. These
two profiling protocols must not be treated as interchangeable. Source-counter
sample shares are not wall-time fractions; instrumented thread-cycle sums are
not GPU elapsed time. A citywide restart also runs Compute Sanitizer memcheck.

The report compares observed-state diagnostic predictors with genuinely pre-run
priors. The prior uses deterministic point queues, routes, departures, free-flow
travel, and known signals; it omits spillback, right-on-red, and acceleration.
Seed 1402 is held out. A second split also holds out four geographic regions.
Forecast parameters and ridge penalty are fixed before looking at test scores.
The static partition table is an offline diagnostic replay-load proxy, not an
implemented GPU allocator or a measured allocation speedup.

## Commands

Use a new artifact/network ID for an independent repetition; never overwrite a
frozen milestone. The recorded run uses these commands and fixed-clock wrapper:

```sh
python3 tools/berkeley_state_study.py prepare
/usr/local/cuda/bin/nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17 src/simulator/scalable_junction_simulator.cu -o artifacts/current/berkeley-state-study-20261014/simulator
python3 tools/test_scalable_junctions.py
python3 artifacts/current/berkeley-state-study-20261014/run_fixed_clocks.py
python3 tools/report_berkeley_state_study.py
```

The wrapper stores/restores clocks in a `finally` block, runs full timing first,
then builds and runs replays, then profiles. It never profiles concurrently with
timing. Build flags match the frozen full-simulation reference's fast-math mode;
the earlier isolated Milestone 2A/leader-index experiment used different flags,
so do not directly compare its timings with this study's timings.

Artifacts: `artifacts/current/berkeley-state-study-20261014/`.
Report: `reports/berkeley-state-study-20261014/`.
Research hypothesis and broader gates: `docs/BERKELEY_STATE_ALLOCATION_EXPERIMENT.md`.
