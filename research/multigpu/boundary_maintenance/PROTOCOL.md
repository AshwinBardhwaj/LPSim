# Experiment 1 — incremental boundary-list maintenance (3 November 2026)

Freeze exchange-study mode 3 as reference. Preserve canonical Car payloads,
traffic calculations, fixed ownership maps, stable global-ID ordering, and both
state/reservation exchange stages. No forecast or partition-policy optimization.

Modes (LPSIM_LIST_MAINTENANCE):
0. Full rebuild: persistent CUB sort plus retained-slot gather, every exchange.
1. Incremental: detect exact membership, refresh incoming records in place if
   unchanged; otherwise compact sorted survivors, sort only added IDs, merge,
   and gather canonical state. Deleted IDs disappear. All surviving ghosts are
   refreshed, not frozen. No unsorted slot layout or changed traffic priority.
2. Hybrid: mode 1 except rebuild if added+removed > max(32, old-local-count/10).
   Threshold fixed before timing; no test-data tuning.

Incremental is not constant-cost: membership detection scans local records and
binary-searches incoming IDs, requires a host decision synchronization, and
changed lists still compact/gather survivors. Explicitly charge those costs.
Stable membership does not eliminate state packing, transfer, reservation
exchange, motion indexes or other unchanged stages. Extra buffers: seven ints
per reserved local slot, two counters, scan workspace (plus pinned counters).

Validation: three logical partitions on Jetson, then three physical A100s. Exact
per-timestep owner/ghost/reference equality and existing physical checks. Compare
final hashes and wire bytes against reference; abort on errors. Include a saved
trajectory with queues/connectors as well as constructed active snapshots. Test
hybrid fallback separately if naturally unexercised. Sort/merge ties retain stable
IDs. No new dead-cell/priority approximation.

Timing: Berkeley 8192 and SF 98304 snapshots plus a developed Berkeley trajectory;
fixed paper-balanced and boundary-aware maps, three GPUs. One excluded warmup,
five shuffled paired repeats, 60 original 0.25s steps per execution. Include a
longer 240-step comparison on the developed trajectory where budget permits.
Mode 0 in the new binary must also match the frozen prior binary. Independent
instrumented component traces excluded from timing. Report all results,
complete-step/stage time, maintenance counts, construction costs, extra capacity,
wire bytes, populations, and paired uncertainty. No requirement to select a winner.

Only retain an optimization as a candidate if full-step median improves >=5%,
repeat-level paired interval excludes zero, no correctness failure; report all
workload regressions. This short study does not establish demand generalization.
No new Nsight sample needed to decide this wall-time gate: GPU event component
traces isolate membership/sort/gather work; counter studies can follow a winner.

Budget: cumulative prior posted $7.35904703 of $10. Refresh billing; reserve at
most 720 seconds × $0.003/s + $0.10 ($2.26). No retries or live deployment.
GPU-function timeout 540s, startup 120s, cleanup allowance 60s. Internal matrix
hard deadline 440s, all subprocesses bounded. Archive partial evidence on failure.
