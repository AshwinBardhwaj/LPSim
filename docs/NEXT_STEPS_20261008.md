# Meeting review and next experiments — October 8, 2026

## Assessment

The proposed single-GPU-first direction, controlled signal experiment, larger mixed
network and reproducible milestones are sound. The movement-versus-density claim
is a hypothesis to test, not an established root cause. A milestone should report
confirmation, rejection, or a mixed result without predetermining the outcome.

Corrections against the latest recorded implementation/results:

- Four main San Pablo–University approaches each have two modeled lanes. Neighboring
  streets are not uniformly two lanes. The latest matched 400-trip following run
  completed **139/400**, versus 117/400 for exclusive reservations; 250/400 is not
  supported by these recordings. At 1,600 trips: 137 completed, 312 active, 1,151
  pending. Pending vehicles have not entered the road network.
- The synthetic cycle is 72 s: each axis gets 20 s through/right green and 8 s
  protected-left green; every phase has 3 s yellow and 1 s all-red. It is not
  simply alternating 20 s red and green.
- Right-on-red requires configured permission, a full stop of at least 2 s,
  no conflicting occupied movement, no conflicting green arrival within 4 s,
  and receiving space. It is prohibited during all-red clearance. Thus it is
  explicitly conditional on signal and traffic state; actual signage is unverified.
- The experimental backend uses fixed, lane-continuous routes; it has no dynamic
  road lane changing. The older lane-map backend's turn-triggered lane changes
  should not be attributed to this backend.
- A physical junction groups OSM nodes and contains explicit lane-to-lane
  connector paths. It is not a zero-length single node. Downstream reservations
  already implement keep-clear behavior. Modeling drivers who intentionally or
  accidentally block the box would be a separate behavioral extension.
- The road kernel still runs leader searches and IDM calculations for stopped
  vehicles (`state == 1`). Speed zero does not disable this work. Pending/done
  vehicles skip it. The distinction is road-active versus pending/completed,
  as well as moving versus stopped within the active population.
- The cited 8.5–9% occupancy and ~67% stalls describe sampled road kernels from
  the earlier experiment, not every GPU stage. L1/TEX throughput around 4% is
  not measured DRAM bus utilization; DRAM counters were unavailable. A long
  scoreboard dependency does not identify which cache/memory level supplied data.
- No universal >50% occupancy requirement applies. Small grids, skipped vehicle
  work, dependencies and uneven workload must be separated experimentally.
- Junction processing is now parallel by vehicle for requests/connector movement
  and by junction for arbitration. The 21 arbiters currently occupy one block;
  this remains a limitation. Dense-run median junction-stage time improved
  9.301 → 1.944 s, with identical trajectories; full wall time 16.081 → 9.075 s.

Sources: `src/simulator/junction_simulator.cu`; recorded evidence under
`artifacts/junction_platoons_20261007` and `artifacts/junction_parallel_20261007`.
Historical source snapshots and raw data remain on the Jetson, outside Git.

## Milestone 1 — establish a valid baseline

Freeze the code commit, model/demand hashes, seeds, timestep, 600 s horizon,
power mode, launch geometry and controller mode. Use the current parallel
following backend; retain serial as a correctness/performance reference.
Record requested, pending, on-road stopped, on-road moving, connector and
completed populations separately at every profiling window.

Run 20, 400 and 1,600 trips under:

1. Current 72 s controller.
2. Always-permitted signal movements while retaining conflict arbitration,
   car-following and receiving-space protection (existing mode 2).
3. A straight corridor with no crossing conflicts and sufficient exit capacity,
   providing a cleaner free-flow control.

Do not call case 2 free-flow unless trajectories actually show it: removing
signals can still leave crossing conflicts, downstream queues and demand overload.
Keep routes/departures identical within each paired comparison. Use at least
three unprofiled timing repeats, then collect separate hardware-counter samples
at matched simulation times and record the associated traffic populations.

Deliverables: timings, speed distributions, active/stopped/pending counts,
completion/entry rates, kernel breakdown and updated GPU/warp tables. Gate:
all movement/gap checks pass and the observed regimes match their labels.

## Milestone 2 — distinguish movement, density and workload layout

Do not interpret a speed-versus-utilization scatter plot alone as causal.
Changing signals changes admissions, population and branch paths at the same time.
Use deterministic per-step snapshot microbenchmarks in addition to full runs:

- Hold N requested and N road-active constant; vary stopped fractions (0%, 50%,
  90%, 100%) on valid lane layouts with adequate gaps. Treat constructed snapshots
  as performance tests, not calibrated traffic trajectories.
- Hold the stopped/moving mix and geometry constant; vary active population.
- Hold physical state and active population constant; vary pending/completed
  padding to test launch waste and the cost of scanning inactive records.
- Permute vehicle storage order while preserving stable priority IDs and physical
  state to test warp grouping. Require identical physical results.
- Independently compare current layout, compacted active work and per-lane leader
  indexing. Do not combine optimizations before measuring each one's effect.

Measure elapsed GPU time/step, time/road-active vehicle, total instruction work
where supported, active threads/instruction, occupancy, eligible warps, issue
activity, long-scoreboard stalls, L1/L2 throughput, memory transactions and
register/local-memory use. Add explicit leader-scan and request counts in a
separate instrumented build; avoid contaminating benchmark timing with counters.
Use source-level profiling to locate dependency-heavy loads rather than assuming
all long-scoreboard stalls come from vehicle positions.

Success means identifying which factors explain cost and documenting counter-
evidence. Do not optimize for utilization alone: a faster kernel can use fewer
resources or show a lower utilization percentage. Allocation should follow
measured work per region, not vehicle speed or density alone.

## Milestone 3 — left-turn pocket pilot (separate behavior experiment)

Inspect raw OSM tags and exporter settings before concluding the data is absent.
OSM supports `turn:lanes`; the cached small GraphML currently exposes `lanes` but
no turn-lane tags. Missing cached tags do not prove the original ways lack them.
Preserve the raw source and add a versioned manual lane overlay when necessary.

Start with one approach at San Pablo–University. A 5 m pocket barely fits the
4.5 m vehicle body and does not accommodate a 6.5 m vehicle-plus-gap slot or the
entry transition under current assumptions. Keep 5 m as a boundary/negative test,
not the default operational pocket. Compare illustrative storage lengths such as
15, 30 and 60 m, with a separately specified transition; verify site dimensions
before calling any of them realistic.

Implement finite pocket capacity and safe entry upstream of the intersection;
left turns leave the pocket, through vehicles retain through lanes, and a full
pocket may cause realistic approach spillback. Do not change signal timing at the
same time. Validate empty/full pockets, competing entry requests, through traffic
passing a left queue, downstream blockage and rollback/spacing at the pocket
boundary. Then add congestion-driven lane choice as its own intervention.

Deliverables: lane overlay, pocket visualization, through delay, turn delay,
pocket occupancy/spillback, safety checks and GPU cost. No pocket is implemented
by this planning change.

## Milestone 4 — scale a mixed network on one GPU

Use reproducible tiled synthetic junctions first, then a larger OSM extract.
Sweep junction count and active population separately, covering free-flow,
queued and mixed regions. Measure scaling of each stage, arbitration imbalance,
launch size, memory use and time/vehicle-step. Larger networks may improve
occupancy, but can also increase search and memory costs; utilization gain is not
guaranteed. Preserve a same-demand serial/parallel comparison where feasible.

Only after measured per-region work is available compare static density priors
against dynamic workload-based allocation. Defer multi-GPU migration and boundary
synchronization until this single-GPU baseline is understood.

## Milestone 5 — paper and reproducibility checkpoint

Frame the paper around safe parallelism, synchronization cost, workload-dependent
GPU efficiency and validated traffic behavior. Describe traffic-flow simulation
and throughput explicitly; do not claim optimal traffic flow without an actual
control/optimization objective and baseline. Incident/CARLA work can remain a
separate project scope rather than an unsupported capability comparison.

Every milestone includes commit/model/demand hashes, environment and power settings,
exact commands, seeds, raw metrics, validation outcomes, limitations and a concise
human-reviewed interpretation. Keep large traces/reports under `artifacts/` and
`reports/`, with small evidence summaries in version control. Publish the next
summary after the hypothesis tests, whether they confirm or reject the hypothesis.

## References

- NVIDIA CUDA best practices (occupancy is not performance):
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- NVIDIA Nsight Compute metric and stall definitions:
  https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html
- OSM lane and turn-lane representation:
  https://wiki.openstreetmap.org/wiki/Lanes
  https://wiki.openstreetmap.org/wiki/Key:turn:lanes
