# Leader-index reconstruction experiment

Preserve the previous Berkeley block-imbalance milestone unchanged. Replace only comparator sorting of vehicle IDs with stable CUB radix sorting of materialized (road, lane, finite position) keys. Initial ID order supplies stable tie-breaking; signed zero is normalized. Vehicle storage and physics remain unchanged. Scratch is allocated once and reused.

Gate: exact sorted IDs, leaders, post-motion state and post-junction state/reservations. Include independent CPU ordering tests and repeated changing state/lane/road fixtures. Replay all 36 matched snapshots (three populations, three demand patterns, four times), with three process batches, three warmups and ten measured repetitions per method in randomized order. Report preparation, sorting, linking and complete pipeline/wall time. Require repeatable complete-step improvement; retain comparator fallback. Full 600-second trajectories must match frozen baseline output before default enablement. No scheduling changes.

Finite positions, road IDs below 2^29 and lane IDs 0..7 are the packed-key contract. These are backend constraints, not new traffic-model assumptions. Allocation is outside per-step timings; persistent extra storage must be reported. Snapshot repeated timings are not full-run throughput claims.

Build control: inspect the frozen simulator PTX; it uses fast-math/FTZ instructions. Use `--use_fast_math` consistently for the final simulator and both replay methods. An initial precise-math build produced small floating-point deviations from the archive while its own comparator/radix trajectories matched exactly. Preserve that diagnostic separately and exclude it from final timing/validation tables.
