# Reproduce exact request gating and guarded multi-rate experiments

Use the immutable `berkeley-block-imbalance-20261019` input milestone and the frozen compact-motion baseline. Production simulator sources are unchanged. The generated standalone experimental simulator includes the same motion and junction equations with isolated experimental branches.

1. Run `python3 tools/build_gating_multirate.py`.
2. Build `tools/gating_multirate_simulator.cu` with `/usr/local/cuda/bin/nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17`, writing the binary to `OUTPUT/simulator` in a new writable output directory.
3. Run `python3 tools/experiment_gating_multirate.py --phase validate --output OUTPUT`. This creates exact reference traces, two-second oracle candidate unions and nine full-horizon validations per variant.
4. Run `python3 tools/experiment_gating_multirate.py --phase timing --output OUTPUT`. This performs a full-horizon warmup and three shuffled timing batches, without validation instrumentation; each final state is checked against its validated variant.
5. Run `MPLCONFIGDIR=/tmp/lpsim-mpl python3 tools/report_gating_multirate.py --input OUTPUT --output REPORT`.

Drivers temporarily fix clocks using `sudo -n jetson_clocks`, record the original configuration and restore it in `finally`. Do not compile or run other GPU experiments concurrently with timing. Existing clock files are protected; use a fresh output directory for repetition.

`trace` is the instrumented baseline. `current` compacts actually eligible request keys before sorting. `oracle` applies offline candidate lists but still recomputes signal/conflict eligibility. `multi2` and `multi4` hold acceleration between two/four base steps only while the guard remains valid; they intentionally approximate the original model. Do not use oracle masks with modified demand or with approximate trajectories.

Per-case `command.json` records exact environment settings. Snapshot/trajectory equivalence applies only to exact variants. The report separately measures approximate state/route divergence, matched-container position/speed errors and completion changes. All timing measurements are full simulations; CPU output and diagnostics are excluded from calculation time, while total subprocess elapsed is reported separately. The original host pre-population counting loop remains part of process elapsed.
