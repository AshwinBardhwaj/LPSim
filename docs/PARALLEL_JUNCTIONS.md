# Parallel GPU junction processing

The physical-junction backend now defaults to parallel scheduling. Set
`LPSIM_JUNCTION_PARALLEL=0` for the retained serial reference. This option is
independent of `LPSIM_JUNCTION_EXCLUSIVE`, which controls traffic policy.
Neither option adds left-turn pockets or congestion-driven lane changes.

## Execution stages

After the existing one-thread-per-vehicle road update:

1. `prepareJunctions`: one thread per vehicle advances connector travel and
   commits completed crossings. Threads also initialize the small summary arrays.
2. `summarizeJunctions`: one thread per vehicle contributes minimum receiving
   positions, approaching-green arrival times, connector tails, reservation
   counts, and the lowest eligible pending vehicle ID for each entry lane.
   Integer atomics and positive-float atomic minima make these reductions
   independent of thread scheduling.
3. `requestJunctionEntry`: one thread per vehicle evaluates its signal, full-stop
   eligibility and conflicts against existing traffic. It produces a request;
   it does not independently grant itself contested resources.
4. `arbitrateJunctions`: one thread per physical junction resolves only that
   junction's competing requests, keeping ascending vehicle-ID priority and
   green-before-red ordering. Existing conflicts were evaluated in parallel;
   the arbiter checks newly granted movements and commits storage reservations.
5. `spawnJunctionVehicles`: one thread per lane admits its preselected pending
   vehicle only after arbitration has published the final exit reservations.

Kernel boundaries on the same stream provide ordering. They are essential:
removing them without an equivalent global synchronization would reintroduce
snapshot and reservation races. No thread spins waiting for another block.

## Ownership and determinism

Every incoming movement must end at its owning junction, and each receiving
road must begin there. Cross-junction conflicts are rejected by input validation.
Thus every mutable movement/receiving reservation has exactly one arbiter writer.
Each car is written only by its own vehicle thread or its owning arbiter in the
respective stage. Spawning selects at most one pending car per lane, using the
same vehicle-ID order as the serial implementation.

Within an intersection admission remains ordered; unrelated intersections run
concurrently. The current 21-junction map launches one block of 128 arbiter
threads (21 useful threads), so arbitration alone does not fill all SMs. Vehicle
stages distribute work across blocks. This is not a claim that all safety-critical
commits can safely execute independently for every vehicle.

## Validation and measurement

Run `python3 tools/test_parallel_junctions.py` on a CUDA device. It executes the
11 existing scenarios under serial/parallel scheduling and following/exclusive
policies. It compares state, movement-event, signal and aggregate trajectory
CSVs byte-for-byte and compares behavioral metrics. Per-step gap, crossing-boundary,
conflict and stopped-in-junction checks remain enabled in both implementations.

Results are kept under `artifacts/junction_parallel_20261007`, separate from
previous experiments. The October 7 platoon experiment's original run/benchmark/
profile tools explicitly select serial scheduling to retain the original
measurement method. In particular, its profiler's launch offsets assume two
kernels per timestep and must not be reused for the new six-kernel pipeline.

Intersection-stage CUDA event time includes all five new kernels and their
intervening launch gaps; total event time also includes road propagation.
Small workloads can lose time to additional launches. Full wall time includes
CPU validation, trajectories and logging and is reported separately.
