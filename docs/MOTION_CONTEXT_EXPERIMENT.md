# Independent pre-motion context experiments

The completed pre-optimization study is commit `80ba153`; its archived outputs are unchanged.

The opt-in scalable backend accepts `LPSIM_MOTION_CONTEXT`:

- `0` (default): original global connector and receiving scans.
- `1`: incoming-lane connector linked lists.
- `2`: shared per-movement receiving/conflict availability.
- `3`: both.

`tools/motion_context.cuh` allocates persistent scratch buffers and rebuilds them
from the immutable `old` vehicle array before every motion launch. CUDA default-stream
ordering separates initialization, collection, resolution and readers. Atomic list
insertion order is arbitrary; minimum-gap ties retain the lowest original vehicle
index, reproducing ascending-ID scan semantics. Equal gaps with the existing road
leader retain that road leader. Receiving blockage uses the original strict
`pos < SPACING + 8` condition. Conflict CSR direction, identical-movement following,
exclusive mode, signals and final-road handling retain original predicates.

Availability is resolved once for each queried movement, using receiving-lane
blockage and occupied connector movements. A dense movement launch skips unqueried
entries. This is intentionally a separate measured implementation choice; no
claim is made that dense movement launches are optimal for every network.

Neither implementation changes junction advancement, request arbitration,
receiving reservations, spawn rules, or right-on-red decisions. Context must not
be reused across timesteps or substituted with post-motion junction summaries.
Scratch memory construction and CUDA memsets are included in replay elapsed time;
one-time allocation is outside the measured per-step interval.

Build with the same flags as the study:

```sh
nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17 tools/berkeley_replay.cu -o artifacts/current/berkeley-motion-context-20261014/replay
nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17 src/simulator/scalable_junction_simulator.cu -o artifacts/current/berkeley-motion-context-20261014/simulator
python3 tools/experiment_motion_context.py
python3 tools/report_motion_context.py
```

The experiment runs 36 saved windows × four variants, 11 small scenarios × three
optimized variants, and 12 full Berkeley scenarios × three optimized variants.
Exact physical CSV/digest comparison is against frozen pre-optimization outputs.
Counter and SourceCounters sampling are separate from timing, at matched 180 s
windows for distributed and concentrated 2,048-trip cases. Baseline counters are
recollected with the same binary/protocol. Full-run executions are correctness
checks, not statistically repeated end-to-end performance benchmarks.
