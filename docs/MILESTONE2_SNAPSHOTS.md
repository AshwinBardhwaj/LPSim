# Milestone 2A: unchanged road-kernel diagnostics

This experiment preserves Milestone 1 and implements no leader indexing,
compaction or data-layout optimization. It isolates movement, population,
inactive padding and storage order in the production `advance` CUDA kernel.

The existing Berkeley input has 5,775 directed roads. Its numerical road geometry
is retained in `data/networks/berkeley_snapshot_reference/model.txt`. Fixtures
use lane 0 and one connected outgoing edge per eligible road, with 20 m spacing,
0 or 5 m/s input speeds, and at least 30 m clearance from road endpoints.
They are constructed performance tests, not calibrated traffic trajectories.

The current junction pipeline supports at most 128 roads and 512 movements;
therefore it is not run on this full network. These results concern road
propagation only. No connector occupants, stop-line admission requests or
crossing conflicts are present. Request eligibility counts are zero by design;
they do not measure the cost of real intersection arbitration.

There are 15 cases: four stopped fractions with 2,048 active/requested vehicles;
256/512/1,024 active populations plus the 2,048 reference; pending/completed
padding to 4,096/8,192 records; and shuffled, lane-sorted, interleaved and
state-grouped permutations of the same 4,096-record physical state.
Stable IDs are retained externally and outputs are compared in canonical order.
Admission-priority permutation remains a separate future validation task.

Each timing repeat performs three warmups and 20 launches against an immutable
input buffer. Output is never fed back. Three independent process repeats use
alternating case order. CUDA events exclude setup, copies and validation.
This measures repeated-state warm-cache work, not full-run throughput. Nsight
runs are separate, with cache-control and clock-control disabled; system clocks
were fixed at the limits of the existing 15 W mode and restored afterward.
The original DVFS pass is preserved but excluded from reported results.

The normal harness directly includes the production CUDA source. A separate
generated instrumented build adds device counters to that kernel. Its output
must match the normal build, but its timing is never used. Instrumented counts
verify A*N receiving checks, A*(N-1) leader candidates and A*N connector checks.

The source commit, source/model/network hashes, binary hashes, compiler resource
logs, original clock settings, raw timings, Nsight reports and CUDA/SASS source
exports are preserved in the artifact directory. Nsight's requested local-memory
allocation metric was unavailable; the compiler reports a 432-byte stack frame,
40 registers and zero register spill loads/stores for advance. Actual local
memory sector counts are reported independently.

Commands (use a new isolated workspace or new artifact ID for a fresh experiment):

```
python3 tools/milestone2_snapshots.py prepare
python3 tools/milestone2_snapshots.py build
python3 tools/milestone2_snapshots.py run
python3 tools/milestone2_snapshots.py profile
python3 tools/milestone2_snapshots.py source
python3 tools/validate_milestone2_snapshots.py
python3 tools/report_milestone2_snapshots.py
```

The fixed-clock orchestration script is retained with the artifact; it saves and
restores settings in a finally block. Do not run timings and profiling concurrently.
Report location: `reports/milestone2-snapshots-20261009/`.
