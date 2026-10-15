# Berkeley state-cost study and static allocation hypothesis

## Order and scope

Keep the frozen Milestone 1 and 2A artifacts unchanged. First compare a lane
leader index against identical Milestone 2A snapshots. Rebuild the index from
each step's input, and charge initialization, sorting, linking and temporary
allocation/synchronization costs to the indexed variant. Change neither Car
storage nor receiving-space/connector scans. Check physical outputs against the
frozen reference before interpreting performance. This is an isolated road-kernel
experiment, not yet a production integration or full Berkeley simulation.

The subsequent full Berkeley experiment requires removing the current junction
pipeline's 128-road/512-movement limits. Merely increasing those constants is
not sufficient: use network-sized lane/movement arrays and per-junction sparse
movement/conflict lists, with deterministic priorities and receiving-lane
ownership. Validate against the serial reference on the existing small network
and adversarial intersecting movements before scaling. The snapshot reference's
synthetic one-outgoing-movement road model is not a full intersection model.
Build a separate Berkeley network with legal lane-to-lane movements, geometry,
signal assumptions, reachable routes, boundary exits and explicit provenance.

## Questions and controlled comparisons

1. Do state-specific costs remain after leader indexing? Compare moving road
   interiors, stopped queues, approaching leaders, stop-line requests, admitted
   connectors, downstream-blocked requests, conflicting/yielding requests, and
   pending/completed records. These labels overlap: retain separate flags and
   counters rather than forcing every vehicle into one exclusive category.
2. Which costs come from search population, topology, and early-exit position?
   Pair snapshots at fixed active population and lane geometry; vary one valid
   state or blocker placement. Changing a stop-line request also changes the
   executed branch, which is an intended intervention and must be documented.
3. Which costs come from workload placement? Hold physical state constant and
   permute storage with stable priority IDs. Compare sustained block count,
   tail duration and eligible warps; do not equate stalled warps with absent
   resident warps.
4. Does a regional cost model generalize? Hold out departure seeds, demand
   patterns, time windows, and geographic regions. Validate on an independent
   network before claiming generality beyond Berkeley.

## Full-run matrix and sampling

Use a 600 s horizon and at least three timing repeats. Freeze source, network,
demand, seed, timestep, controller, GPU clocks/power and launch geometry.
Choose demand levels in a pilot to produce observed uncongested, mixed, and
saturated regimes; retain identical routes/departures for paired controller and
backend comparisons. Do not reuse San Pablo trip counts as citywide regime labels.
Include the current signal controller and always-permitted movements with all
conflict and receiving-space protections retained.

Save restartable full device-state snapshots at 60, 180, 450 s and selected queue
formation/discharge events. Capture requested, future departures, pending,
road-stopped, road-moving, connector and completed counts. Record regional lane
counts, queue lengths, headways, approach distances, signal phases, request counts,
grant/denial reasons, leaders and available receiving space. Measure state
transitions across each window as well as instantaneous counts.

Collect unprofiled end-to-end GPU and host times separately from Nsight counters.
Profile matched snapshots and source lines with restored identical inputs.
Report index construction, motion, request preparation, arbitration, spawning,
and any dispatch/transfer costs separately. For each kernel report duration,
registers/local-memory traffic, achieved versus theoretical occupancy, grid and
block sizes, eligible warps, active threads/instruction, issue activity,
scoreboard stalls, instruction work, L1/L2 traffic and throughput. Where a
counter is unsupported, mark it unavailable. Instrument scans and requests in
a separate build and verify its outputs match the timing build.

Whole-kernel counters alone cannot identify a vehicle state's contribution.
Use controlled counterfactual replays and source-correlated samples. State-only
work lists alter occupancy and cache behavior; report those as diagnostic
microbenchmarks, not additive estimates of the original full kernel's time.
Likewise, regional replays must include required neighbor/connector state and
explicitly account for changed launch size and cache locality.

## Falsifiable hypothesis and static policy

Hypothesis: after search optimization, regional cost is better predicted by
expected road-active population, intersection request/connector activity,
receiving-space checks and local conflict degree than by density or mean speed
alone. Congestion can increase or decrease particular terms. Do not assume a
coefficient's sign before measuring it.

At initialization, derive a prior over state populations from known departures,
routes, road lengths/lane capacity, free-flow travel times, signal schedules and
historical demand if available. An offline coarse queue model may forecast
arrival and service windows; LWR/ARZ are optional, not required. No held-out
future vehicle trajectories or measured test-run timing may enter this prior.

Fit nonnegative cost components where appropriate, with explicit interaction
terms for conflict degree and requests, plus uncertainty. Compare predicted
regional work over the whole horizon and over coarse time bins. Freeze one
partition before the run (static policy); a precomputed changing schedule is a
separate policy, not the same static partition.

The GPU allocator must have a concrete actuator: region-to-work-queue/block
assignment or tile sizes, not a claim to reserve SMs through ordinary launches.
Partition to balance predicted work while penalizing cross-region dependencies,
data movement and long-running tails. Retain synchronization and deterministic
intersection ownership. Include prediction, partition construction, indexing and
dispatch costs in the end-to-end comparison.

Baselines: equal geography, equal expected vehicles, topology-only weights,
measured training-run costs, and a separately labeled hindsight oracle.
Acceptance requires faster total runtime with identical physical results, not
merely higher utilization. Reject the hypothesis if traffic features add little
held-out predictive value beyond active population/topology, or if scheduling
overhead exceeds saved work. Report negative results and prediction error.

## Deliverables and gates

- Isolated leader-index timings and Nsight tables, including full rebuild cost.
- Validated full Berkeley model and scalable serial/parallel junction parity.
- Full-run table joined to per-window state populations; controlled replay table
  separating state effects from population and storage effects.
- Source-level dependency evidence, counter-evidence, and proposed cost terms.
- Held-out prediction results and a static policy specification before any claim
  of allocation benefit; then an end-to-end policy ablation.

Each stage gets a new artifact/report directory with immutable inputs, hashes,
commands, validation, raw counters, figures and limitations. No failed validity
gate is relabeled as a validated traffic result.
