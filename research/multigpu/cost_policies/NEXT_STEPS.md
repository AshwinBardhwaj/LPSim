# Next experiments: allocation based on measured cost

The aim is to minimize complete simulation time under correctness and device-memory constraints. GPU utilization and long-scoreboard percentages are diagnostic features, not the optimization objective. The existing results justify more informative candidates, but do not yet demonstrate net multi-GPU acceleration or a useful learned prior.

## Evidence that determines the next step

The frozen local-scaling study uses three A100s on one host, not three networked clusters. At 98,304 initially active SF vehicles, median times for 60 original timesteps are 29.313 ms (original one GPU), 47.535 ms (local one GPU), 185.769 ms (best measured two-GPU policy), and 216.028 ms (best measured three-GPU policy). The last two are approximately 6.34 and 7.37 times slower than the original single GPU. These are short 15-s workload tests, not full 600-s runs or calibrated traffic demand.

For three GPUs, balanced partitioning sends 20.128 MB and takes 216.028 ms; the community map sends only 11.267 MB but takes 220.481 ms. Balanced also reduces observed active max/mean imbalance from geography's 1.628 to 1.124. Bytes and vehicle balance each explain only part of the tradeoff. The differences between close policies require more timing repeats before firm ranking.

On the largest balanced three-GPU case, average step costs are:

| Stage | ms/step |
|---|---:|
| Motion/indexes | 0.378 |
| First state exchange | 1.138 |
| Junctions | 0.503 |
| Reservation exchange | 0.357 |
| Spawn | 0.101 |
| Second state exchange | 1.107 |
| Total | 3.584 |

Exchange/reservation stages account for 72.6%, but include packing, host count handling, full-record copying, stable-ID sorting/gathering and barriers. They are not measurements of bare NVLink latency. Even accounting away all three stages leaves roughly 0.98 ms/step, versus about 0.49 ms/step for the original single-GPU median. This subtraction is an accounting estimate, not a measured counterfactual: changing execution can change the remaining stage times. It explains why a better partition alone is unlikely to erase the current gap.

The earlier hardware microbenchmarks measured similar isolated peer bandwidth on all six directed pairs (~263 GB/s for 64-MiB transfers); small-copy plus host-sync latency was ~20 microseconds. The large-copy result does not predict our small batched exchanges, and the allocation can change between Modal jobs. GPU-ID permutation is a low-priority hypothesis unless new hardware discovery finds meaningful asymmetry.

Sources: `reports/local-scaling-20261028/{timing-summary,allocation-summary,stage-means}.csv`; `reports/modal-hardware-20261027/README.md`. Older Nsight kernel counters describe a different backend and must not be relabeled as counters for this one.

## Recommended policies, in order

1. **Calibrated work-and-boundary partitioning.** Balance predicted regional motion/index, request, connector and topology work; penalize repeated ghost refresh, receiving-space dependencies and ownership churn. Score the slowest rank at each synchronization stage. Start from the paper maps and the two new proxy candidates, then refine local boundaries. Keep junction arbitration and its reservations under one authoritative owner. A queue's residence time can make a low-flow cut expensive even when few vehicles cross it. The new `boundary_aware` implementation tests a first approximation of this mechanism; the calibrated cost model and boundary refinement remain to be built.

2. **Partitioning for useful communication overlap.** Prefer boundaries that leave substantial genuinely independent interior computation while boundary messages are in flight. Compare this against ordinary work/byte balancing with the same asynchronous execution path. Need an explicit dependency graph: receiving-space and conflict effects can cross a boundary even without a migrating car. Streams/events can express dependencies, but neither API availability nor the prior hardware GEMM overlap result proves traffic work can hide these exchanges. This mechanism requires implementation and an optimistic schedule test before a predictor.

3. **Forecasted, migration-penalized boundary changes.** Forecast work and boundary state over several signal cycles; move a boundary only when conservative integrated savings repay forecasting, partition construction, list/pool reconstruction, transfers and synchronization. Compare against fixed partitions and a cheap current-state trigger. Add hysteresis/minimum residence time and retain the current partition when uncertain. The pure amortization gate is implemented; online authority-transfer/repartition execution is not. No stale ghosts or relaxed simulation timesteps are permitted.

4. **Forecasted pool capacity and GPU count.** Size owned/ghost/send pools from upper-tail local demand, with explicit headroom and safe overflow handling. Select one/two/three GPUs based on net runtime and memory need. Currently the candidate backend has no online pool growth, and excess A100 memory was not shown to be a bottleneck; do not claim memory-allocation speedup without allocation stalls or a capacity problem. At the current tested scale, choosing one GPU remains the performance baseline to beat.

## One bounded measurement round before learned policies

Freeze all current artifacts. Add a separate diagnostic build with per-rank CUDA events and host spans for:

- Packing/retention, host count copies and worker wakeups.
- Peer transfer, distinguishing send time from waiting for a late sender.
- Stable-ID key construction, sorting, gathering and junction-summary refresh.
- Motion/index, junction/request, reservation and spawn stages.
- Owned road/connector/pending/completed populations; ghost count, retained records, migrated records, cut lanes, nonempty peer messages, actual payload size and pool occupancy.

Use an Nsight Systems timeline if available, or equivalent recorded events. Collect Nsight Compute samples separately at matched states for motion, index/list maintenance and junction kernels: duration, achieved occupancy, SM throughput, active threads/instruction, eligible warps, long-scoreboard samples and memory traffic. Source attribution must identify expensive loads in this backend; a high stall fraction in a very short kernel is not by itself a wall-time bottleneck. Instrumentation stays out of timing runs.

First repeat the old assignments with each transport change independently. Candidates are persistent stable-ID lookup, refreshing only dynamic state for existing ghosts (retaining routes), and replacing host-wide waits with dependency events where correctness allows it. They address framework cost; compare every allocator on the identical resulting backend. Do not credit these implementation improvements to a traffic prior.

## Cost model and validation

For a synchronized backend, score a candidate as:

```
step_cost = sum over stages(max over GPUs(predicted stage duration))
            + additional unhidden host/global cost
run_cost  = sum over future steps(step_cost)
            + prediction + construction + switching costs
```

Include each cost exactly once. Per-stage durations must include exposed transport/list costs; do not add another blanket communication penalty on top. On an overlapped backend, use the dependency graph's critical path instead of summing stages that overlap. Account for concurrent-link contention, not isolated peak bandwidth. Memory limits are hard constraints. The global-sized topology arrays and fixed worker/barrier costs remain visible; the current map cannot eliminate them.

Fit a simple nonnegative stage model or piecewise model on training runs only. Include owned/visible records, index sizes, requests, connector counts, boundary residents, migrations, messages, and topology summaries. Report prediction residuals and feature ablations. Aggregate stage totals from the existing six cases cannot identify reliable per-region coefficients: we need the per-rank/substage data above. Exclude future occupancy and evaluated timings from a claimed pre-run prior.

First choose with measured future costs (explicitly labeled an oracle candidate), evaluate in independent repetitions, then replace those costs with forecasts available before the run. Freeze demand seeds and complete held-out runs before fitting; adjacent timesteps from the same trajectory are not independent train/test examples. Retain the existing synthetic-demand families and add bursty/concentrated demand on the larger topology. Measure actual active population, not requested demand alone.

Compare original one GPU, local one GPU, geography, both paper-style policies, the two new candidates, the measured-cost candidate, and a cheap current-state rule. Use common routes, departures, snapshots and physics, exact physical/ownership checks, an excluded warm-up, and at least five randomized unprofiled timing repetitions per configuration. Validate physical multi-device correctness before timing each new map. Report cold and explicitly amortized steady-run costs separately, plus cost per simulation and absolute wall time.

**Preregistered progression gate for the next study:** at least 10% lower aggregate complete-run time than the strongest baseline selected using training/validation only, after all relevant overhead, with a paired 95% interval excluding zero improvement across held-out demand runs and no correctness failures. Report individual workloads and regressions. If the budget cannot support independent held-out runs, label the outcome exploratory and do not declare the gate passed. Separately report whether any multi-GPU policy beats the original single GPU; relative improvement over a slow multi-GPU baseline is not strong scaling.

## Contribution and budget

The baseline paper already uses demand-weighted partitioning and formulates a compute/communication min-max objective. Merely adding traffic weights or minimizing the slowest GPU is not a demonstrated new contribution. The candidate research contribution is an evidence-backed forecast of *time-dependent boundary residency, critical-path work and persistence*, used to make decisions that beat fixed traffic-weighted partitions and simple reactive rules on held-out workloads, with all transition costs charged. Novelty still needs a broader literature review.

Paper: [LPSim, section 3.3](https://arxiv.org/html/2406.08496v1#S3.SS3). CUDA dependency/peer-copy semantics: [NVIDIA multi-GPU programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/multi-gpu-systems.html).

This turn used no Modal compute. Last recorded cumulative study spend is $6.2797 of the existing $10 authorization, leaving about $3.72. Recheck posted billing and reserve the full timeout before another job. Use the remainder for one targeted diagnostic/pilot round, not an expansive held-out performance campaign. Return small result files promptly; compress archives after releasing the GPUs to avoid repeating the previous packaging overhead.
