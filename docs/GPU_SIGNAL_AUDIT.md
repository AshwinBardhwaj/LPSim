# GPU signal audit and minimal repair plan

Scope: ordinary ground vehicles in the current checkout. No traffic dynamics were
changed by the lane visualization work. Routing-in-chunks is out of scope.

## Code findings

- `src/simulator/cuda_traffic_sim.cu`, `kernel_intersectionOneSimulation`, cycles
  incoming approaches with `deltaEvent = 20.0f`; 0x00 means red, 0xFF means green.
  Because it tests `currentTime > nextEvent`, actual transitions are quantized to
  simulation steps and need not occur at exactly 20-second intervals.
- The ordinary-vehicle forward scan in `kernel_trafficSimulation` goes directly
  from the current-edge vehicle search to `// NEXT LINE`. It does not insert a
  red-light obstacle before computing IDM acceleration.
- `nextVehicleIsATrafficLight` is initialized false and never assigned true in
  this CUDA kernel. The lane-change guard using it therefore does not suppress
  changes for an approaching red light.
- Signal values are used in lane-change gap calculations, and a separate UAM
  insertion path reads/writes them. Thus "signals have no effect anywhere" and
  "pure car following only" are inaccurate descriptions of this codebase.
- The CPU `simulateOnePersonCPU` has a `// b) TRAFFIC LIGHT` block that treats a red
  signal as a stationary obstacle, setting the gap, relative speed, found flag,
  and `nextVehicleIsATrafficLight`. This is the local reference for the repair.
- CUDA lane-change attempts require speed above 3 m/s and occur every five
  simulation steps (2.5 seconds with the current 0.5-second timestep). The kernel
  includes gap checks and collision rollback, whose correctness still requires
  controlled tests; a visible lane change alone is not validation.
- CUDA lanes have separate lane-map addresses and explicit lane-index changes.
  Lane 0 is leftmost (left changes decrement the index). At an edge transition,
  an out-of-range lane is clamped to the new edge's last lane. This is not proof
  of a realistic merge negotiation or lane-to-lane intersection connector model.
- Berkeley nodes contain 207 `highway=traffic_signals` tags. The SP lane-map
  initialization creates intersection controllers by topology; it does not
  select those tags. Enforcing its existing signals everywhere would therefore
  create synthetic signalized junctions, not reproduce Berkeley's real signals.

Conclusion: source inspection confirms a missing ordinary-vehicle red-light
braking check in this kernel. It does not quantify violations in an existing
trajectory file, which contains no signal state observations.

## Small, staged change

1. **Controlled fixture first.** Two connected roads with a signalized junction;
   one vehicle, then a platoon. Hold red and green in separate runs using the same
   input and initialization. Save signal state, route index, position and speed.
   Establish the current behavior before changing it.
2. **Restore the existing CPU idea in CUDA.** Between current-road vehicle scan
   and next-road scan, treat a red at the stop line as a zero-speed leader. Use
   `currentEdge_d` (GPU-local lane index), not the global lane-map index. Exclude
   UAM. Select the closest relevant obstacle, use a positive gap epsilon, and set
   `nextVehicleIsATrafficLight` so existing lane-change gating can work.
3. **Prevent numerical red crossings.** Add a stop-line movement bound before
   edge transition, accounting for both displacement calculations in this kernel.
   A coarse timestep must not jump the stop line. Vehicles already past it must
   clear rather than teleport backwards. Keep lane occupancy and speed consistent
   with the bounded position. Do not solve this with an early return that skips
   lane-map writes or migration bookkeeping.
4. **Select real controlled nodes.** Load a small explicit node-ID signal list
   derived from `nodes.csv` tags. Initialize uncontrolled approaches permissively
   and skip their phase updates. Map IDs correctly into each GPU partition.
   Reuse the existing cycle initially, labeled synthetic timing. No new scheduler,
   turning-phase optimizer, yellow phase or real-world timing claims in this pass.
5. **Export signal events.** Emit changes as timestamp, physical incoming edge,
   and red/green state, including initial state. Record the state used by each
   step, so playback and validation use the same timing convention. Add markers
   only from this data, never infer light colors from stopped vehicle dots.
6. **Compare with the preserved baseline.** Run forced-red, forced-green,
   red-to-green queue discharge, blocked downstream, one/two lanes, short edges,
   0.1/0.5/1.0-second steps, and partition-boundary checks where hardware permits.
   Require no red crossings from upstream, no lane/vehicle overlap introduced by
   clamping, queue release on green, and unchanged uncontrolled-link behavior.
   Then run identical Berkeley demand and report completion, travel time and
   queue/delay differences, with runtime overhead.

Likely touch points: CUDA vehicle kernel; SP network/lane-map setup; a small
intersection-control flag/list; optional signal-event exporter; focused fixtures.
Steps 2–3 are a localized kernel repair. Correct geographic signal selection and
validation are necessary additional work; copying the CPU block alone is not a
complete, trustworthy Berkeley signal model.

## Reproduction and presentation

`python tools/package_lane_review.py --output artifacts/NAME` captures the existing
CSV/routes/network, current config/source, converter/viewer and hashes, then builds
lane-aware playback. It does not rerun or alter the simulator. Serve with
`python -m http.server 8080 --directory artifacts/NAME`.

The package contains `rebuild.py` for rebuilding playback without this checkout.
The manifest explicitly distinguishes current packaging provenance from unknown
historical simulation provenance. Future simulation runs should capture their
source/config/build/device versions and random seed before execution, rather than
retroactively claiming these establish the provenance of an older CSV.

## Existing Berkeley lane recording: observed results

The lane-aware conversion accepted all 12,800,594 input observations across
20,000 output trips, with zero duplicate samples and 87,411 lane-index changes
between consecutive observations on the same route edge. No invalid lane index,
position, topology, or kinematic split was reported. This establishes that lane
state changes are present in the recording; it does not validate gap acceptance,
turn permissions, or collision freedom. `playback/manifest.json` identifies the
lane-change CSV so individual cases can be reviewed with the sampled timestamps.
