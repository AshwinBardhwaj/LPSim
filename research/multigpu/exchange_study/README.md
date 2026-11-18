# Exchange framework ablation and measured-cost selection

This study preserves `local_scaling/local.cu` and the original frozen physics.
Its own `local.cu` supports `LPSIM_EXCHANGE_MODE=0..3`:

| Mode | Change |
|---|---|
| 0 | Original full-record retention and Thrust sorting |
| 1 | Persistent CUB radix-sort scratch/output storage, allocated during setup |
| 2 | Retain local slot references, append incoming records, sort global IDs, gather into the other state buffer |
| 3 | Both changes |

Mode 2 avoids the intermediate write of full records for vehicles retained on
their GPU. It still gathers the entire sorted local list. Mode 1 still sorts
every exchange: it removes repeated scratch allocation, not sorting work. Full
peer payloads, counts, reservation protocol, local list reconstruction, immutable
pre-motion state and authoritative ownership remain unchanged. Stable global IDs
preserve priority after local slot permutation. Capacity checks precede transfer
application. These are framework improvements, not traffic-aware decisions.

Compile `local.cu` using the frozen `../source/research/multigpu/partitioned.cu`
closure, CUDA `-O3 --use_fast_math -lineinfo -std=c++17 -Xcompiler -pthread`, and
`sm_87` locally or `sm_80` on A100. The diagnostic binary adds
`-DEXCHANGE_DIAGNOSTICS`; the timing binary has no event/row instrumentation.
`trace.cuh` resolves GPU event intervals at existing barriers. Host call spans,
device synchronization spans and rank completion differences are preserved
separately. None should be mislabeled pure network latency or idle SM time.

Files:

- `run.py`: physical correctness, independent backend ablations, training-only
  backend selection, five common-backend policies, and independent test timings.
- `profile.py`: optional separate Nsight kernel replay at matched initial state;
  logs counter permissions/failures rather than substituting old measurements.
- `modal_study.py`: bounded single-use three-A100 function; small compressed
  output tables/hashes instead of duplicate full-state archives.
- `launch.py`: cumulative spending guard and client watchdog. It refuses to
  overwrite a launch reservation. Failed attempts and charges remain preserved.
- `analyze.py`: SHA-256 verification, per-GPU component/wait tables, allocation
  metrics, selector charges, figures and PDF. It requires a passing cloud result.
- `PROTOCOL.md`: preregistered workload, selection and interpretation rules.

Reproduction inputs and outputs are under `artifacts/exchange-study-20261029`;
reports are under `reports/exchange-study-20261029`. The old studies are unchanged.
The initial cloud attempt passed its first physical check but failed in the
Python collector because backend-specific metric dictionaries had duplicate
keyword names. The collector was corrected; that attempt is retained as
`attempt1`, excluded from timing evidence and included in spending.

This pilot uses repeated timings of two fixed workloads, not held-out demand
patterns. Charging all oracle trial runs describes the implemented measurement
procedure; it is not a lower bound on a future inexpensive predictor's cost.
Forecasting needs evidence of a useful optimistic opportunity and subsequent
held-out net benefit, as specified in `cost_policies/NEXT_STEPS.md`.
