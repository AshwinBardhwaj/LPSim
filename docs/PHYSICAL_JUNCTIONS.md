# Physical-junction experiment mode

`LPSIM_JUNCTION_SCENARIO=/absolute/path/scenario.txt build/lpsim` selects the
opt-in CUDA physical-junction backend in `src/simulator/junction_simulator.cu`.
The existing lane-map simulator remains the default. This mode is an explicit
experiment implementation, not a retrofit silently applied to old recordings.

The backend now defaults to the [parallel junction pipeline](PARALLEL_JUNCTIONS.md).
The commands below reproduce the earlier platoon-policy comparison and explicitly
select serial junction scheduling to preserve its profiling method.

Preparation requires Python packages `networkx` and `shapely`; report generation
also requires `matplotlib`. The cached OSM source GraphML is included with the
small network, so preparation does not require an Overpass download.

Prepare the small OSM network and cases:

```
python3 tools/prepare_junction_experiment.py
cmake -S . -B build
cmake --build build -j2
python3 tools/run_junction_experiment.py
python3 tools/run_junction_experiment.py --exclusive
python3 tools/validate_junction_experiment.py
python3 tools/benchmark_junction_experiment.py
python3 tools/profile_junction_experiment.py
python3 tools/report_junction_experiment.py
```

Network inputs and derived junction geometry are under
`data/networks/san_pablo_lane_preserving`. The preparer groups connected OSM
nodes linked by roads shorter than 25 m, including the four San Pablo–University
nodes. External road geometry is cropped 8 m before/after the junction nodes;
lane connectors bridge those endpoints. This is schematic, not surveyed geometry.
Tiny roundabout clusters in this reduced map are also treated as controlled
junctions; actual roundabout yielding is not modeled.

Normal demand uses designated outer-network groups, not junction interiors.
Diagnostic fixtures seed cars on entry/exit roads outside connector interiors.
Lanes are assigned from the planned turn: leftmost for left, rightmost for right.
There is no discretionary lane-changing model in this mode.

The synthetic 72 s controller alternates NS and EW through/right green (20 s)
and protected left green (8 s), each followed by 3 s yellow and 1 s all red.
No new entry on yellow; admitted cars clear independently of subsequent signals.
Red rights require a completed 2 s stop, configured permission, no conflicting
occupied swept corridor, no conflicting green arrival within 4 s, and exit space.
Permissions are experiment assumptions, not verified local signs. No pedestrian
or cyclist calls are present; the experiments do not validate those interactions.

Straight connectors preserve the incoming lane index. Left turns are leftmost-to-
leftmost and right turns are rightmost-to-rightmost. Unequal lane counts permit
only common through indices; no lane-change or merge connector is introduced.
The 132 legal connectors define lane-continuous routes, with no mid-road lane
changes in this experimental backend. Consequently demand differs from October 6.

Conflicting movements retain exclusive swept-corridor and exit-lane ownership.
Identical movements may follow with at least 2 s admission headway and 6.5 m
spacing. Every in-flight vehicle reserves one receiving-road storage slot:
minimum(nearest receiving vehicle position, road length) must be at least
6.5 × (in-flight reservations + 1) + 8 m before granting another entry.
This prevents assuming that several cars can all use the same vacant exit slot.
The reservation count persists until each connector vehicle reaches the road.
`LPSIM_JUNCTION_EXCLUSIVE=1` restores exclusive same-movement admission for a
matched baseline. Both modes use the same generated network, demand and safety
checks. As before, this is conservative conflict management, not field calibration.

CUDA parallel road propagation uses the previous vehicle snapshot and the prior
IDM parameter values, a 4.5 m vehicle, a 2 m minimum gap, and a hard spacing bound.
Green unobstructed approaches need not stop. Turns traverse connectors at 4 m/s;
through movements at 7 m/s. The retained serial reference arbitrates on one GPU thread. The default pipeline
uses per-vehicle request threads and concurrent per-junction arbiters; see
`PARALLEL_JUNCTIONS.md`. This is intended for small
networks (up to 128 roads, 4 lanes each, 512 movements), not a new GPU scaling claim.
Only one GPU is used. Its performance is not comparable to the old lane-map mode.

Runtime invariants are checked after every 0.25 s step: finite bounded positions,
nonnegative speeds, at least 6.5 m spacing on roads, connectors and across their boundaries; no simultaneous conflicting
connectors, no stopped connector vehicles. Entry events are checked against the
movement permissions and full-stop requirement. Fixture tests additionally cover
red through/left holds, green completion, red right turns, yielding and blocked
exit refusal. Recorded states are exported every 0.5 s.

Outputs are in `artifacts/junction_platoons_20261007`; report and playback are in
`reports/san-pablo-platoons-20261007`. All earlier experiments remain intact.

Signal-structure references (the timings here are not calibrated from them):
- https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter4.htm
- https://ops.fhwa.dot.gov/publications/fhwahop08024/chapter5.htm

## GPU measurement scope

Nsight Compute 2024.3.1 is invoked with passwordless sudo to access hardware
counters, sampling two launches of each kernel at 60, 180 and 450 simulated
seconds for each of the three trip counts and both controllers. Isolated profiler
folders contain replay reports, raw CSVs, commands and binary hashes. The report
includes achieved/theoretical occupancy, active threads per instruction, eligible
warps, issue activity, stalls, L1/TEX and L2 utilization, and launch resources.
DRAM throughput was absent from the device metric catalog; no substitute value
is claimed. NVIDIA definitions: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html

Three separate unprofiled runs per configuration record CUDA event intervals,
host step-and-synchronize time and full process wall time. Profile replay times
are never mixed into those timing medians. Road propagation scans all vehicles
for every road vehicle; its worst-case work is quadratic. The serial reference used for this historical comparison remains a single GPU
thread; the later parallel comparison is documented separately. This comparison does not test a CPU
baseline, static partitioning, or the original lane-map backend.
