# Exchange framework and measured-cost selector — preregistration

Preserve the prior local-scaling baseline and all physics. New experiment modes:
0 original exchange; 1 persistent CUB radix-sort workspace; 2 slot-reference
retention (avoid writing all retained full records); 3 both. Stable global-ID
sorting and authoritative reservations remain mandatory. All policies use the
same selected execution path. Do not count framework speedup as policy gain.

Separate compiled diagnostic and uninstrumented timing binaries. Diagnostics
record per-rank GPU event intervals and host call intervals for pack, count D2H,
peer copies, key construction, sorting, gathering, junction refresh, reservation
packing/unpacking, motion/index, junction and spawn. Events do not introduce
additional within-stage waits; resolve at the existing synchronization points.
Host cudaDeviceSynchronize duration includes outstanding GPU execution, not
pure idle time. Rank completion spread at a barrier is a diagnostic proxy, not
a measurement of idle SMs. Event intervals can include host submission gaps.

Frozen Berkeley 8,192 and SF 98,304 active snapshots, 0.25-s timestep, original
72-s controller, 60 steps (15 simulated seconds), 20-step exact-reference gates.
Source/input hashes accompany results. This is a short workload experiment,
not calibrated traffic or a full 600-s benchmark. Validate new maps on physical
devices before timing. Stop on failed physical/ghost/authority checks or hash
mismatch. Instrumented timings are excluded from performance comparisons.

Training: excluded warm-up plus three shuffled timing repetitions. Choose a
single backend using mean normalized training runtime across the two workloads:
at least 5% mean improvement, no workload >5% slower; otherwise retain mode 0.
Evaluate all four modes in five separate timing repetitions. Also retain the
original dense single-GPU and unmodified local multi-GPU implementations.

Compare geography, paper-balanced, paper-community, work-balanced and
boundary-aware maps on two and three GPUs through the chosen backend. Select
the measured-cost candidate from three training repetitions for each workload
and GPU count; assess that frozen choice on five independent randomized timing
repetitions. Report an optimistic pre-supplied selection, and charge all trial
runs, lookup, partition construction and setup separately. Baselines are selected
using training only. Report construction on Jetson as such, not deployment-host
timing. Data sharing/replicated topology, memory capacities, pending/completed
records and all unchanged stages remain charged.

These are independent timing samples, not unseen demand patterns. The forecasting
gate in cost_policies/NEXT_STEPS.md requires >=10% net held-out complete-run gain,
paired uncertainty excluding zero, and exact correctness. This small study can
reject a weak opportunity or justify a later held-out study, but cannot establish
that generalization gate by using repeated timings of these two workloads.

Budget: last posted cumulative $6.27972814 of $10; refresh billing before launch.
One three-A100-80GB function, 900-s timeout, no retries, 120-s startup allowance,
60-s return allowance. Reserve 1,080 seconds at a conservative $0.003/s = $3.24,
including CPU/memory margin, plus $0.10 artifact allowance. Do not launch if the
ledger plus reservation exceeds $10. Internal matrix deadline 730 s; return small
tables/hashes, delete duplicate final-state files after checks; gzip level 1.
No forecasting/training model or long-lived deployment in this experiment.
