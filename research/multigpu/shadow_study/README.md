# Shadow-mode partition pilot

Protocol: PROTOCOL.md. No ownership map is changed during the capture run.
Snapshots at t=180/300/420 are replayed for 60 seconds with fixed, reactive and
causal-forecast maps. Candidate state/cost inputs end at the decision time.

- `local.cu`: unchanged prior-study backend plus optional checkpoint capture.
- `propose.py`: direct snapshot decode, current rank-cost attribution, bounded
  traffic trend, weighted METIS, ownership-label alignment and transfer counts.
- `validate_local.py`: every-step original-reference and uninterrupted-replay check.
- `run.py`: one fixed capture/case; immutable topology reuse; five shuffled repeats;
  reference gates, continuity hashes, and separate diagnostic replays.
- `modal_study.py`, `launch.py`: single bounded cloud job and budget reservation.
- `analyze.py RAW_RUN_DIRECTORY`, then `report.py`: tables, break-even migration
  budgets, forecast accuracy and PDF. Python needs numpy/pandas/matplotlib;
  proposals additionally need pymetis 2023.1.1.

`LPSIM_CAPTURE=1` enables 30-second checkpoints. `LPSIM_EXCHANGE_MODE=3` selects
the validated full-rebuild backend. `LPSIM_LOGICAL=1` maps all ranks to Jetson
GPU 0 for local correctness testing, not physical multi-GPU performance claims.

Do not relaunch without refreshing the billing ledger. The current $15 ceiling
is cumulative; source launch reservations deliberately fail if reused.

Migration payload is only a vehicle-record lower bound. Actual switching also
requires ghost/index/reservation handling and synchronization. Only a positive
remaining budget can justify implementing and measuring that switching path.
Negative conservative results may still leave an opportunity for a cheaper
observer or generator; replay-only results show that distinction explicitly.
