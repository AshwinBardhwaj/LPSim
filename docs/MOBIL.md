# MOBIL lane changes in the physical-junction simulator

Enable with `LPSIM_MOBIL=1`. The default is off. Supported paths are the current
scalable physical-junction simulator and the GPU-resident multi-GPU experiment.
The legacy `cuda_traffic_sim.cu` lane-grid heuristics and archived baselines are
separate implementations and are not converted by this option.

The model uses IDM acceleration predictions and full MOBIL incentive:

    ego acceleration gain
      + politeness * (old follower acceleration gain + new follower acceleration gain)
      > threshold

The target follower's predicted acceleration must be at least `-safeBraking`.
Front and rear target gaps must also preserve the simulator's 6.5 m center-position
spacing. Acceleration predictions use the physical backend's IDM parameters,
vehicle length and signal stop line. Junction conflict/storage checks continue in
the normal junction pipeline; MOBIL does not replace them.

| Environment variable | Default | Meaning |
|---|---:|---|
| `LPSIM_MOBIL` | `0` | Explicitly enable with `1` |
| `LPSIM_MOBIL_POLITENESS` | `0.3` | Weight for follower effects, range 0–1 |
| `LPSIM_MOBIL_SAFE_BRAKING` | `4.0` | Maximum imposed target-follower braking, m/s² |
| `LPSIM_MOBIL_THRESHOLD` | `0.2` | Positive incentive threshold, m/s² |
| `LPSIM_MOBIL_JUNCTION_MARGIN` | `20.0` | Excluded distance at both road ends, meters; at least 6.5 |

These are configurable initial values, not calibrated behavior for a particular
city. No approximate/skipped state refresh is used.

## Parallel execution and traffic policy

1. Finish motion, junction processing and spawning, producing a stable snapshot.
2. Build a GPU lane/position index. Each candidate finds its current and adjacent
   lane leaders/followers by binary search and evaluates both adjacent lanes.
3. Each vehicle submits its best legal proposal. Equal scores prefer the lower
   lane index. An integer atomic minimum chooses the lowest global vehicle ID
   among all proposals on each road.
4. Commit at most one lane change per road per timestep. Independent roads commit
   concurrently, avoiding overlapping-gap and simultaneous-swap races without
   serializing the whole network.
5. Rebuild normal motion indexes from the resulting snapshot on the next step.

The one-change-per-road restriction is deliberately conservative. It can limit
lane-change throughput on long, busy roads; it is an explicit scheduling policy,
not part of the MOBIL formula. A future compatible-gap scheduler must preserve
determinism and revalidate simultaneous changes before relaxing it.

This first implementation covers discretionary mid-road changes. It does not
implement mandatory merge urgency, continuous lateral trajectories, lane changes
inside connectors, or changes in the junction exclusion zone.

## Routes and multi-GPU ownership

`Car::lanes[k]` records the actual current lane; `Car::moves[k]` is updated to a
legal connector with that incoming lane, the same next road, and the same planned
next-road lane. Future route segments are unchanged. If no such connector exists,
the candidate is rejected. Consequently, lane-preserving maps can legitimately
permit few changes on non-final roads. Historical route entries are not rewritten
to imply that the car entered its current road in its new lane.

The resident backend keeps every lane of a road under one owner. Only that owner
proposes changes; ghosts are read-only. The entire owned road population is local,
so neighbor queries and arbitration need no remote decision maker.

After the existing post-spawn exchange, each owner runs MOBIL. A third, separately
versioned **lane-change exchange** publishes full Car state before any neighbor's
next motion/context build. This transmits the changed lane and connector, retains
buffer-consumption dependencies, and forbids ownership migration in that phase.
Disabled mode allocates only the original two exchange buffers and launches no
MOBIL kernels. The three-phase sequence namespace reserves distinct epoch values.

Archived dense multi-GPU transports carry only a compact immutable-route payload;
they do not support these mutable lane/connector fields. Use the resident backend
for MOBIL, not those archived transports. `evaluate.py` rejects enabled MOBIL
against those old-model baselines.

## Validation and use

```bash
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_87 \
  tests/test_mobil.cu -o /tmp/test-mobil
/tmp/test-mobil

bash research/multigpu/gpu_resident_exchange/build.sh /tmp/mobil-build sm_87
python3 tests/test_mobil_integration.py --binary /tmp/mobil-build/resident \
  --output /tmp/mobil-logical-validation --logical
```

Use `sm_80` for A100 and omit `--logical` on a host with three peer-accessible GPUs.
The integration suite checks enabled and disabled operation across 1/2/3 partitions,
both schedules and both slot layouts, compares each step with the serial MOBIL
reference plus frozen motion/junction physics, requires real lane changes, and
compares final hashes and accepted-change counts. Logical partitions validate the
algorithm but do not establish physical NVLink visibility or performance.

The serial MOBIL reference uses brute-force neighbor searches and serial winner
selection; it shares scalar equations, not GPU indexing or arbitration. Unit cases
cover overtaking, unsafe followers, target overlap, politeness, connector legality,
stable-ID permutations, ghost ownership, independent roads and disabled behavior.

Existing same-physics timing figures do not apply to MOBIL. Enabling it adds lane
index/proposal/arbitration work and, on the resident backend, a full state refresh.
Do not claim a speedup or fidelity calibration without matched experiments.

Model reference: https://mtreiber.de/MicroApplet/MOBIL.html

### Recorded validation (2026-11-18)

On the local Orin GPU, all 24 integration configurations passed, covering 1,920
exact timestep comparisons. Each enabled run accepted four changes and produced
the same final-state hash across 1/2/3 logical partitions; disabled runs accepted
none and also agreed. The receipt is
`artifacts/mobil-20261118/logical-validation.json`. CUDA compilation also passed
for the A100 (`sm_80`) resident target. Physical multi-GPU execution and performance
have not been measured.
