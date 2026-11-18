# Incremental boundary-list experiment

Experiment 1, 3 November 2026. Read PROTOCOL.md before interpreting results.
The frozen physics closure is artifacts/multigpu-preparation-20261027/source.
The baseline is exchange_study mode 3; no existing production backend is changed.

- `LPSIM_EXCHANGE_MODE=3` for every run.
- `LPSIM_LIST_MAINTENANCE=0`: full sorted-ID rebuild/gather.
- `LPSIM_LIST_MAINTENANCE=1`: scanned membership detection, unchanged-list refresh,
  otherwise survivor compaction + addition sort + stable merge + gather.
- `LPSIM_LIST_MAINTENANCE=2`: same with fixed high-churn fallback.
- `LPSIM_FORCE_REBUILD=1`: verification-only forced fallback for changed lists.

Detection still visits local records. This is event-driven *list repair*, not
constant-cost event production from the physics kernels. Unchanged lists avoid
sort/gather, but retain required fresh ghost/owner payload exchanges. Extra
synchronization and scratch are included. The hybrid threshold was fixed before
running: additions + removals > max(32, old count / 10).

Local compile (place source closure as ../source relative to an include dir):
```
nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_87 -Xcompiler -pthread \
  -I/tmp/lpsim-local-scaling/study local.cu -o /tmp/lpsim-boundary-maintenance
```
For A100 use sm_80. Add `-DEXCHANGE_DIAGNOSTICS` only for component traces, never
for timing comparisons. The Modal image uses the frozen CUDA container digest.
`prepare_inputs.py` freezes existing maps/snapshots; no new demand is generated.
`launch.py` enforces the remaining authorized budget. `run.py` verifies physical
states, ghost equality, stable IDs, authority uniqueness, final hashes and wire
bytes, then performs shuffled paired timing repeats. `analyze.py RAW_DIR`
produces the PDF, figures and machine-readable metrics.

Artifacts: artifacts/boundary-maintenance-20261103/
Reports: reports/boundary-maintenance-20261103/
