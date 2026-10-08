# Parallel GPU junction execution — October 7, 2026

The default physical-junction backend now evaluates connector travel and entry requests with one GPU thread per vehicle. Concurrent per-junction arbiters retain deterministic green-before-red, vehicle-ID-ordered commits. Entry-lane spawning is parallel by lane.

## Correctness

All 11 scenarios were run under both following and exclusive traffic policies, producing 22 serial/parallel pairs. State, movement-event, signal and aggregate CSVs matched byte-for-byte. All per-step spacing, cross-boundary spacing, conflict and stopped-in-junction checks passed. Three additional paired timing repetitions per demand also matched. An invalid cross-junction conflict topology was correctly rejected.

CUDA Compute Sanitizer memcheck also passed the 80-second queued-platoon fixture with zero errors when run with sudo. The ordinary-user attempt could not enable GPU debugging; its log is retained.

## Unprofiled timing

Three repetitions; medians for each 600 simulated seconds / 2,400 steps. Same binary, inputs and following policy, alternating execution order. GPU event timing excludes CPU validation/output; wall time includes them.

| Trips | Serial junction s | Parallel junction s | Junction speedup | Serial total GPU s | Parallel total GPU s | Serial wall s | Parallel wall s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.233 | 0.187 | 1.25× | 0.347 | 0.302 | 1.081 | 1.078 |
| 400 | 2.271 | 0.962 | 2.36× | 3.345 | 2.486 | 5.082 | 4.081 |
| 1600 | 9.301 | 1.944 | 4.79× | 13.802 | 6.491 | 16.081 | 9.075 |

The five-kernel intersection pipeline is timed as a whole, including launch gaps. Some tiny controlled fixtures are slower because the extra launches outweigh useful parallel work. The road kernel is unchanged; full-run speedup is smaller than junction-stage speedup.

## Hardware-counter sample

Nsight Compute, sudo, 1,600 trips at 180 s; two complete steps in each mode. Serial offsets use 2 kernels/step; parallel offsets use 6. These replayed samples are not the unprofiled timing benchmark. Full counter exports and commands are in `profiles/`.

| Mode | Kernel | Block | Grid | Mean duration µs | Active threads / instruction |
|---|---|---|---|---:|---:|
| serial | advance | (128, 1, 1) | (13, 1, 1) | 4200.77 | 7.08 |
| serial | junctionStep | (1, 1, 1) | (1, 1, 1) | 11416.30 | 1.00 |
| parallel | advance | (128, 1, 1) | (13, 1, 1) | 4207.49 | 7.08 |
| parallel | arbitrateJunctions | (128, 1, 1) | (1, 1, 1) | 2275.12 | 20.84 |
| parallel | prepareJunctions | (128, 1, 1) | (13, 1, 1) | 25.65 | 31.16 |
| parallel | requestJunctionEntry | (128, 1, 1) | (13, 1, 1) | 140.85 | 8.91 |
| parallel | spawnJunctionVehicles | (128, 1, 1) | (2, 1, 1) | 18.37 | 27.48 |
| parallel | summarizeJunctions | (128, 1, 1) | (13, 1, 1) | 39.44 | 16.49 |

## Limits and use

Arbitration inside each junction remains ordered. On this small map its 21 arbiters fit in one 128-thread block; that stage alone does not distribute work over all SMs. The parallel per-vehicle stages use 13 blocks at 1,600 trips. Road propagation still scans all vehicles for leader detection. Left-turn pockets and discretionary lane changes are not included in this scheduling change.

Set `LPSIM_JUNCTION_PARALLEL=0` to use the serial reference. `LPSIM_JUNCTION_EXCLUSIVE` independently chooses the traffic policy. The older platoon report tools pin serial scheduling to preserve their two-kernel profiler methodology.

Reproduce: `python3 tools/test_parallel_junctions.py`; repeat timings with `--case network_20 --case network_400 --case network_1600 --repeat 3 --exclusive 0`. Profile with `python3 tools/profile_parallel_junctions.py`. Source design: `docs/PARALLEL_JUNCTIONS.md`.

Existing playback remains behaviorally applicable because the trajectories match exactly. Prior report files have not been rewritten.
