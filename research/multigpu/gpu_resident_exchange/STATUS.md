# Implementation status — 2026-11-05

The CUDA P2P prototype is implemented and passes local logical-partition
correctness checks. The full implementation plan is **not complete**: NVSHMEM,
physical peer validation, broad interior overlap and scaling measurements remain.

## Implemented

- Frozen baseline/source manifests and dependency/priority audit.
- Device-resident counts and GPU-side peer reads; no normal-path count readback.
- Persistent local Car slots and global-ID lookup; refresh without full Car
  sorting/gathering, plus a selectable rebuild ablation.
- Global-ID adapters for connector ties, leader ordering, junction arbitration
  and pending spawn selection.
- Epoch-validated state batches, bounded capacity errors, explicit publication
  and consumption events, and neighbor-only data dependencies.
- Independent local-lane spawning ahead of remote reservation dependencies.
- Exact-gated evaluation against mode 3 and both single-GPU baselines; paired
  timing analysis that rejects logical-partition runs as scaling evidence.

## Validation completed

The local GPU is Orin. GPU execution requires access outside the filesystem
sandbox. CUDA compilation and logical-rank execution do not establish physical
NVLink correctness.

| Case | Logical partitions | Steps | Result |
|---|---:|---:|---|
| Berkeley, 8,192 vehicles | 2 | 60 | All six variants agree |
| Berkeley, 8,192 vehicles | 3 | 60 | All six variants agree |
| Developed Berkeley, 2,048 vehicles | 3 | 240 | All six variants agree |

The three new variants performed 1,080 exact-reference timestep comparisons in
this final matrix. Every visible ghost is compared, unique authority and physical
constraints are checked, and final hashes match both single-GPU baselines.

GPU protocol checks passed for empty messages, insertion, refresh, removal, reuse,
duplicates, overflow, stale epochs and invalid ownership phase. Six CPU tooling
tests passed. Compute Sanitizer could not run its checks because GPU debugging
features are disabled; this is not a sanitizer-clean result.

Evidence: `artifacts/gpu-resident-exchange-20261105/findings.json`, the three
`final-*/receipt.json` files, their per-variant logs and outputs, and `build-source/`.
Verified-run wall times include reference work and are not performance evidence.

## Next required work

1. Supply an existing two/three-GPU host; verify peer access, physical delayed
   consumer/buffer reuse, exact state and protocol progress before timing.
2. Integrate NVSHMEM with a pinned compatible SDK and explicit PE/process launch
   model. The present transport is CUDA P2P, not NVSHMEM.
3. Measure the ablations with fixed maps and matched hardware. Account for the
   new capacity grids and comparator sorts; they may consume transport savings.
4. Use traces to justify further splitting of interior motion/junction work and
   persistent radix workspaces. Current neighbor scheduling still has host launch
   submission and possible library-internal synchronization.
5. Claim useful scaling only after beating both single-GPU baselines on matched
   workloads. No speedup or default-backend promotion is claimed at this stage.
