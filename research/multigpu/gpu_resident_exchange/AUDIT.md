# Dependency and implementation audit

Date: 2026-11-05. Baseline is exchange-study mode 3, included from a frozen
physics closure by `prepare.py`; current production files are not used or changed.

## Baseline costs and dependencies

| Site | Host dependency / work | Replacement in experimental CUDA path |
|---|---|---|
| `Exchange::state` pack | Local-record scan; retained keys; outgoing full Car records | Classify/pack local capacity; update authority and refresh epochs in same kernel |
| count D2H | Counts copied to pinned host memory, then device/Team synchronization | Envelope count stays on GPU; destination kernel reads peer count |
| transfer | CPU reads sizes and computes destination offsets | Per-neighbor fixed-capacity outboxes, GPU pull into persistent slots |
| list rebuild | Sort and gather full local list after both exchanges | Lookup/refresh existing slots; allocate new IDs; retire stale ghosts |
| `Local::resize` | Host sets `n` for motion/context/request indexes | Fixed allocated-capacity launches with inactive slots; capacity overhead remains |
| reservation exchange | Pack, host rendezvous, peer copy, unpack, rendezvous | Publish event; peer-reading unpack kernel; consumed event protects reuse |
| `Team::run` | CPU workers rendezvous after phases | Host enqueues neighbor CUDA event waits; optional barrier mode retained |

The new CUDA path is **not NVSHMEM**. It uses CUDA peer-addressable memory and
device kernels to consume variable-size messages. CPU launch submission remains;
message-count readback does not. No remote spin waits are used. NVSHMEM integration
requires a compatible SDK and a process/PE launch design, which this environment
does not currently provide. Do not describe the backend as fully GPU-autonomous.

## Identity assumptions audited

- Road leader order: frozen radix sort is stable in ascending local slot order;
  baseline keeps slots sorted by global ID. New comparator sorts by lane,
  position, global ID. Comparator versus radix cost must be reported.
- Compact motion: connector ties compare slot indices. Generated adapter changes
  only the tie comparison to global IDs, preserving arithmetic expressions.
- Pending spawn: summary uses atomicMin(local slot), with `n` as sentinel. Adapter
  uses global IDs and INT_MAX; spawn translates back through ID-to-slot lookup.
- Junction requests: key low bits are slot index and double as priority. Adapter
  uses global ID; arbitration translates through lookup before the unchanged
  admission function.
- Connector list insertion order is nondeterministic. Global-ID tie-breaking
  remains necessary; list traversal order alone must not decide a tie.
- Snapshot gathering/verification must gather by global ID, not physical slot.
- Position indexes still rebuild each motion step. Persistent membership does
  not imply persistent leader order.

## Phase graph and buffer lifetime

1. Motion and connector exits finish locally before state phase 0 publication.
2. Destination waits on source's published event, installs state, retires stale
   ghosts, then records consumption. Junction summary follows installation.
3. Junction arbitration completes before publishing reservations. Zero values
   are included. Only lanes owned by the local upstream junction may spawn before
   remote reservation dependencies; boundary lanes wait.
4. Spawn finishes before state phase 1 publication. Ownership changes in this
   phase are errors. Next motion follows completed local phase 1 installation.
5. Before reusing a per-phase source outbox next step, wait for consumption by all
   topology neighbors. Host enqueue order records producers before consumers;
   CUDA event waits capture the intended recorded event generation.

Neighbor sets are the undirected cross-owner road graph, including empty roads.
This includes receiving ghosts and possible connector-exit migrations. Reservation
dependencies are directed. Published event and envelope epoch jointly establish
visibility and version; ordinary flag polling is not used.

## Remaining limitations / required gates

- The sandbox hides the local Orin GPU. Escalated execution supports logical-rank
  correctness tests; there are no two/three physical GPUs on this host. Compilation
  and logical tests do not establish peer visibility, progress, or speedup.
- NVSHMEM is not installed; no NVSHMEM backend has been compiled or integrated.
- Physical multi-GPU delayed-consumer and long-run reuse testing remains necessary.
- Compute Sanitizer reports that GPU debugging features are disabled on this Orin;
  no sanitizer-clean claim is made.
- Local Car capacity is bounded (default twice initial visible population, capped
  at global population). During exchange, obsolete ghosts are retired after
  incoming installs; transient capacity can exceed final live population. Explicit
  capacity override or a clean overflow failure is provided; no online growth.
- ID lookup is a replicated integer array of global population size; Car storage
  is local. Capacity grids, retirement scans, and ordinary motion indexes remain.
- Thrust comparator/request sorts may allocate and synchronize internally despite
  the no-sync execution policy. Removing explicit barriers does not prove overlap.
- Interior spawn can run before remote reservation dependencies. Interior motion
  and split junction pipelines are not implemented; they need a broader summary
  and index dependency split after physical exactness gates pass.
- No transport-only implementation preserving the exact old radix/dense-local
  layout is provided yet. `rebuild-barrier` versus `slots-barrier` isolates optional
  full-record rebuilding within the new backend, not every change from mode 3.
- No per-peer ownership generation field is implemented separately from the
  phase epoch; ordered channels and single-phase ownership transfer constrain
  this prototype. Asynchronous migration beyond that schedule is unsupported.

## Success criteria

Primary project result: complete-step time beating the original single GPU and
the local-framework single GPU on matched workloads, with 2/3-GPU scaling reported.
Intermediate gains versus the old multi-GPU backend are useful evidence, not project
success. No 5% threshold is treated as a scaling target or promised result.
