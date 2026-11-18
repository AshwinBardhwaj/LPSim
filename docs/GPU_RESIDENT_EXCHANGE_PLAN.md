# GPU-resident boundary exchange implementation plan

Status: implementation in progress, 2026-11-05. The isolated CUDA P2P backend,
dependency audit, persistent slots, neighbor event schedule and exact-gated runner
are in `research/multigpu/gpu_resident_exchange/`. Local Orin logical-partition
validation is available; physical multi-GPU performance has not been measured.
NVSHMEM integration remains outstanding. See that directory's AUDIT.md for scope
and limitations rather than interpreting the milestones below as completed.

## Objective and baseline

Reduce complete-timestep cost while preserving the existing exact traffic
semantics, fixed ownership maps, timestep, and reservation authority. Replace
CPU-mediated exchange scheduling with persistent storage and GPU-side publication.
Keep the two state-exchange phases and reservation exchange initially.

Baseline: `research/multigpu/exchange_study/local.cu`, exchange mode 3, with its
frozen physics source closure. Preserve the original single-GPU and local-framework
single-GPU baselines. The November incremental-maintenance candidate is an ablation,
not the default: its detection scans and host decision barrier erased savings.

Create an isolated experiment under `research/multigpu/gpu_resident_exchange/`.
Do not overwrite previous source closures, reports, or artifacts. Freeze source,
input, ownership-map, compiler and dependency hashes before comparisons.

## Required consistency contract

- Exactly one authoritative writer per vehicle and one authority per junction's
  admissions/reservations. Storage location must not determine priority.
- Messages identify source, destination, stable vehicle ID, timestep, phase and
  ownership generation. Reject stale, duplicate and wrong-phase transitions.
- Preserve post-motion exchange before dependent junction work, authoritative
  reservation exchange before dependent spawning, and post-spawn ghost refresh
  before the next dependent motion step.
- Membership changes are distinct from state refresh. An unchanged ghost still
  receives fresh state. Publish zero reservations and empty-batch completion.
- A migration revokes the source's right to execute that vehicle for the next
  dependent phase and activates the destination only after installation. An old
  copy may remain a ghost, never a second authority.
- Each directed neighbor channel has bounded buffers, a publication sequence and
  a consumed sequence. Reuse requires receiver acknowledgement, not just local
  transfer completion. Double buffering alone does not prove safe reuse.
- Payload and header must be visible before readiness is published. Use the chosen
  transport's documented ordering and completion operations; ordinary stores and
  a volatile flag are not a sufficient protocol.
- Capacity overflow or invalid protocol state must stop safely and surface an
  error; peers must not wait forever for a success signal that will never arrive.

## Milestone 0 — freeze dependencies and establish observability

Deliver `PROTOCOL.md`, build/source manifest, and a dependency audit.

1. Map `Local`, `Exchange::state`, `Exchange::reservations`, `Team::run`, and the
   timestep loop to their reads/writes, launches, count readbacks and barriers.
2. Audit the included frozen `Partition` kernels and indexes. Record every use
   of array position as priority, contiguous live records, host-side `n`, sorted
   global IDs and shared summary initialization. Include leader tie-breaking,
   junction requests, spawn selection, snapshots and validation.
3. Build directed neighbor sets from topology, including state, migration and
   reservation dependencies; do not infer a permanent neighbor set from currently
   populated ghosts alone.
4. Inspect target A100 peer access and installed CUDA/NVSHMEM versions. Pin a
   compatible version; isolate process/rank setup from the existing CPU Team API.
5. Specify measurements: complete-step wall time, launches, count D2H operations,
   host barriers, records scanned/sorted/gathered, bytes by message class, memory
   capacity, and producer-to-consumer delay. Keep diagnostics outside timing runs.

Exit: reproducible mode-3 baseline and a complete dependency/priority inventory.

## Milestone 1 — transport prototype with existing phase barriers

First choice: NVSHMEM put-with-signal and GPU-side waits, with persistent symmetric
inboxes and device-resident message counts. NCCL device LSA is a fallback candidate,
not a second implementation to build simultaneously. Keep the CUDA peer-copy
baseline available for comparison.

Deliver `transport.cuh`, `protocol.cuh`, a transport-only harness and build support.

1. Allocate per-neighbor, per-phase bounded inboxes, headers, readiness/consumption
   sequences and error state once. Document symmetric allocation and capacity cost.
2. Pack using existing semantics, then publish count plus payload from the GPU.
   Start with a separate publication kernel after packing completes; fuse only
   after proving cross-block completion and memory ordering.
3. Receive based on device counts without a host count round trip. Use bounded
   grids with device-side count guards where possible. Explicitly address any
   library sort that requires a host item count: a fixed-capacity masked sort is
   an acceptable charged prototype, not the final performance solution.
4. Retain the existing phase barriers to isolate transport effects. Convert the
   fixed-size reservation channel first, then variable-size state channels.
5. Follow NVSHMEM's launch/progress requirements for synchronizing kernels. Avoid
   arbitrary spin kernels that can occupy resources needed by a producer.

Tests: empty and maximum-size batches, delayed producer/consumer, consecutive
epochs, slot reuse, stale/duplicate headers, overflow, zero reservations and
error propagation. Test on physical GPUs; logical ranks cannot establish peer
visibility or GPU-side progress correctness.

Exit: exact simulation results with no normal-path message-count D2H readback.
Report transport timing independently of remaining list-maintenance costs.

## Milestone 2 — persistent slots and event-produced membership changes

Deliver `slot_store.cuh`, `membership.cuh` and audited index adapters.

1. Maintain stable-ID-to-slot lookup, slot generations, authority/ghost metadata
   and device-resident live counts. Reserve capacity before stepping; define an
   explicit quiescent growth/fallback path or bounded failure.
2. Initially retain canonical full Car payloads. Install refreshes in existing
   slots; allocate/release only on membership transitions. Do not add payload
   compression to the same experiment.
3. Produce membership transitions while existing packing/classification work
   determines current ownership and ghost destinations. Compare against stored
   prior membership there, rather than adding a separate full detection scan.
4. Preserve stable global-ID priority through explicit ordered index views or
   global-ID keys. Remove assumptions that physical slot order equals priority.
   Charge lookup, index maintenance and masked-capacity work in timings.
5. Motion/leader indexes still reflect changing positions every step; persistent
   membership does not justify retaining a stale position index.
6. Keep a full-rebuild reference path for debugging and measured fallback. Never
   silently change storage policy midway through a timed run.

Exit: unchanged membership performs no membership sort/full-Car gather; all
refreshes and migrations match the exact oracle. Include churn-heavy cases and
stable-ID ties, not only low-churn snapshots.

## Milestone 3 — neighbor dependencies and interior overlap

Deliver `schedule.cuh` and a written per-stage dependency graph.

1. Replace each global phase barrier individually with producer publication and
   consumer waits on required neighbor epochs. Retain semantic phase ordering.
2. Classify independent work from the complete read dependency graph, not merely
   distance from a partition edge. Account for connectors, receiving-space tests,
   lane leaders, summaries and reservations.
3. Start with local junction work during post-motion exchange and local spawning
   during reservation exchange where dependencies permit. Then evaluate interior
   next-step motion during post-spawn refresh.
4. Separate boundary and independent work on streams/queues; prevent shared
   summary resets or buffer writes from racing. Transfer completion is insufficient
   if installation/index preparation has not finished.
5. Preserve within-junction priority and arithmetic order. Do not let a partition
   advance arbitrarily far ahead; bounded channels and consumed epochs constrain
   progress. Prove absence of circular waits for the scheduled phase graph.

Exit: delayed-neighbor stress tests pass and traces show useful work overlapping
communication. No per-exchange CPU rendezvous remains in the optimized path;
host orchestration, optional diagnostics and final completion may remain.

## Milestone 4 — evaluate and select

Compare mode 3; transport only; persistent slots only where practical; combined
transport/slots; and combined plus overlap. Hold ownership maps and physics fixed
to distinguish execution-engine gains from allocation-policy gains.

- Use 1, 2 and 3 GPUs. Include Berkeley 8,192, SF 98,304 and the saved developed
  trajectory with queues/connectors. Add zero-boundary, high-churn and skewed-work
  cases for correctness and diagnosis. Longer and larger timing cases follow only
  after correctness and capacity gates pass.
- Compare every authoritative vehicle and ghost with the frozen oracle each step
  in verification runs; preserve physical gap/conflict, unique authority and
  reservation checks. Gather by stable ID for canonical comparison.
- Use excluded warmups and at least five shuffled paired timing repetitions.
  Keep verification, tracing and profiling out of timed runs. Report setup and
  full-process cost separately; include all communication work in timestep time.
- Report actual active population, ghost population, bytes, membership churn,
  slots/capacities, GPU count and topology. Do not sum overlapping event spans as
  wall time or label host synchronization duration pure NVLink latency.
- Optional intermediate promotion gate: >=5% complete-step median improvement over mode 3,
  paired repeat-level interval excluding zero, no correctness failure, and no
  target workload >5% slower for a universal default. Retain workload-specific
  candidates explicitly if they do not meet the universal gate.
- Separately report whether any multi-GPU candidate beats both single-GPU
  baselines. Relative multi-GPU improvement is not proof of useful strong scaling.
  Primary project success requires beating the single-GPU baseline on matched
  workloads; a 5% improvement over a slow multi-GPU implementation does not satisfy
  that objective. No target speedup is promised in advance.

## Scope and execution order

Implement milestones sequentially, retaining selectable intermediate variants.
Do not combine this work with new partition policies, forecasting, changed
timesteps, stale ghosts, approximate physics or altered traffic priority.

Complete local implementation and available checks before scheduling physical-GPU
experiments. Historical cloud budgets are not a fresh budget for this plan; prepare
a bounded run matrix and cost estimate before any newly authorized cloud launch.

Final deliverables: protocol, isolated backend, deterministic validation harness,
reproducible source closure, ablation CSVs, complete-step comparison and a report
that records negative results as well as improvements.

## Technical references

- NVSHMEM signaling: https://docs.nvidia.com/nvshmem/api/latest/gen/api/signal.html
- NVSHMEM CUDA interactions: https://docs.nvidia.com/nvshmem/api/latest/cuda-interactions.html
- NCCL device API: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html
- Related GPU-initiated halo exchange: https://arxiv.org/abs/2509.21527
