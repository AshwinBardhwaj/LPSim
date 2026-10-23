# Memory policy experiment reproduction

Baseline commit: `89cb7c2f023e206c4719e83db34dd85fd9e01a0b`.
Inputs: immutable `berkeley-block-imbalance-20261019` archive and Berkeley junction network. Workloads and time windows are defined in the drivers. Use a new writable output directory; archived outputs must not be overwritten. Run GPU benchmarks alone, with no compilation or other GPU workloads in parallel.

Build each replay with `/usr/local/cuda/bin/nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17`:

- `tools/memory_storage_replay.cu` → OUTPUT/storage_replay
- `tools/memory_index_replay.cu` → OUTPUT/index_replay
- `tools/shared_memory_replay.cu` → OUTPUT/shared_replay

Generated headers are retained; `tools/build_memory_variants.py` and `tools/build_shared_soa.py` reproduce them from the compact baseline. The production simulator's compact-state option is opt-in. Mandatory correctness indexes and reservation logic remain unchanged.

Run sequentially:

1. `python3 tools/experiment_memory_storage.py --stage storage --output OUTPUT` (includes evolving chains).
2. `python3 tools/experiment_memory_indices.py --phase characterize --output OUTPUT`.
3. `python3 tools/fit_memory_policy.py --output OUTPUT` (freeze configurations before evaluation).
4. `python3 tools/experiment_memory_indices.py --phase evaluate --output OUTPUT`.
5. `python3 tools/experiment_memory_storage.py --stage shared_final --no-chain --output OUTPUT`.
6. `python3 tools/report_memory_policy.py --input OUTPUT --output REPORT`.

Drivers set and restore Jetson clocks using passwordless sudo, preserve their prior clock configuration and reject overwriting it. The retained historical study keeps evolving-chain checks in `storage_validation` and a clean timing rerun in `storage`; for new runs, the report accepts chain records in either directory. The first diagnostic timing run is excluded from final performance tables. Shared-memory final runs include grouping-only, AoS shared and SoA shared controls.

The PDF and CSVs report paired within-process medians, not best timings. The 5% gate requires all three process batches for a given matched window to pass. All methods run the same motion equations and validate exact canonical output and reservation arrays. Policy costs include CPU selection/forecast/map setup under explicit amortization assumptions; they are not a claim of an implemented online adaptive simulator.
