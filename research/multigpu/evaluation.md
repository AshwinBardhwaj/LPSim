# Preregistered evaluation, frozen before A100 results

## Question and baselines

Does a topology/demand prior improve **complete simulation runtime**, including prediction, partitioning, packing, transfers, synchronization and migration, beyond inexpensive partition policies running identical physics and data structures?

The prepared policies share one junction-owner-map interface:

1. `geography`: equal-width east/west strips, assigned using junction coordinates.
2. `vehicle_count`: deterministic largest-weight-first assignment using observed road-active plus connector counts. Pending and completed records do not count as active work. An empty initial state explicitly falls back to unit junction weights; this is not evidence for active-count balancing. Repartition cost must be charged when this policy is refreshed later.
3. `communication_graph`: deterministic local graph-cut refinement, initialized by geography, with a quadratic population-balance penalty. Static edge weights use lane counts as a conservative message-volume proxy. It is a heuristic, not METIS and not an optimal cut. Its objective and 30-pass limit are fixed in prepare.py.

A future traffic-aware prior must beat the strongest applicable measured baseline, not only equal geography. Include the optimized single-A100 implementation as a strong-scaling baseline. Keep incremental indexing independently selectable and use the same choice across policies in each comparison.

## Frozen inputs and split discipline

`workloads.json` fixes 27 fresh cases: train seed 2701, validation 2702, test 2703; distributed, concentrated and bursty departures; 2,048, 8,192 and 16,384 requested trips. Routes are sampled from the existing Berkeley pool with a modest lane/speed preference. The demand is synthetic, not calibrated. Test inputs are frozen but may not be used to fit a prior. Correctness inspection of test trajectories is allowed; performance-based model selection is not.

Historical seed 1901 snapshots at t=0 or t=180 are development/correctness inputs, never newly claimed held-out data. Count-based policies on populated snapshots receive only the supplied current state. Initial empty snapshots use the documented fallback. Future prior training must use only training trajectories and topology/demand known before the evaluated run.

Berkeley has 5,013 roads, 12,216 movements and 1,672 junctions. No larger validated junction network is available locally. Smaller San Pablo networks exist, but a second unseen larger network still must be acquired and validated before cross-network generalization can be claimed. Higher requested demand is not automatically higher active work. Extend to 32k/64k/100k+ only after documenting active road/connector populations, capacity, valid gaps and usable runtime.

## Hardware and correctness gates

Before performance measurements: discover actual devices, memory, power/clock settings, PCIe/NVLink topology, directional peer-access capability and small/large transfer costs. Do not assume NVLink from the A100 label. Probe Nsight Compute counter permissions and Nsight Systems availability separately; record blocked capabilities without substituting invented metrics.

First run one A100, then three logical partitions on that A100, then the identical protocol on three physical A100s. Require exact physical equivalence on the **same device architecture and toolchain**, every timestep. Jetson-versus-A100 bitwise results can differ under fast-math; report cross-architecture differences separately and never relax within-A100 equivalence to hide a partition bug.

Before a performance claim, replace sequential prototype transport with concurrently launched per-device work, explicit CUDA events and supported peer copies (or a measured host fallback). Add timing mode without the reference/comparison copies, verify separately, and include warm-up and all stage costs. Current prototype wall times are diagnostic only.

## Timing and reporting

Use at least 5 unprofiled repeats per configuration after 1 warm-up. Randomize policy order with seed 2710. Use identical demand/controller/timestep/horizon and pinned compiler configuration. Run profilers separately at matched simulation times (30, 180, 360, 540 s) and record pending, stopped-road, moving-road, connector and completed populations, partition loads and migrations there.

Report cold construction and warm steady-state separately, plus total wall time from input preparation through final synchronization. Exclude file serialization from compute-only timing but report it separately. Include policy inference, graph construction, task/pool setup, index maintenance, pack/unpack, transport, boundary waits, ownership migration and capacity growth. Fixed memory budgets include ghosts, metadata, messages and double buffers.

Use Nsight Systems to inspect device overlap, copy traffic and barriers. Use Nsight Compute on representative kernels for duration, occupancy, active threads/instruction, eligible warps, issue activity, long-scoreboard stalls, L1/L2 transactions/throughput, registers/local memory and mapped source samples. No causal claim from speed-versus-utilization scatter alone. A utilization increase is not a success metric.

## Success criterion

Predeclare a **10% geometric-mean reduction in complete-run wall time** over the strongest baseline across held-out cases, with the lower bound of a paired 95% bootstrap confidence interval above zero improvement. Bootstrap cases with repeats nested, not individual steps as independent samples. Require no held-out case with a median regression exceeding 5%, zero correctness violations, and compliance with the same memory budget. Report failure and counterexamples; do not change the gate after seeing A100 results.

First test a measured-cost oracle on separate construction/timing repetitions, including its costs, before fitting a predictor. If the oracle fails this gate, do not train a prior for that mechanism. A predictor that merely reproduces a cheap current-state rule has not demonstrated a value for forecasts.
