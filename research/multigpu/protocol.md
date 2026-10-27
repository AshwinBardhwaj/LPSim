# Three-partition correctness protocol, version 1

This protocol preserves the existing compact-state, following-junction simulation. It does not change signal phases, gap rules, stable priorities, routes, or timesteps. The simulator runs at dt=0.25 s with a 600 s full-run horizon and the current synthetic 72 s controller (mode 0). Modes 1 and 2 are validation controls. Mode 2 is always-permitted signals, not necessarily free flow.

## Ownership

A policy assigns each complete physical junction to exactly one partition. All roads ending at that junction belong to the same partition, including all lanes. Pending vehicles belong to their first road's owner; road vehicles to their current road's owner; connector vehicles to the connector junction's owner. Completed vehicles retain their final road owner. Stable vehicle IDs and routes never change.

Junction admission, conflicting-movement protection and receiving-lane reservations are authoritative at the **upstream junction of the receiving road**. The receiving road itself belongs to its downstream junction. Those authorities can differ. No partition may independently grant a conflicting movement or override a reservation.

Road vehicles are ghosted to their road's upstream junction owner when that differs from their authoritative owner. The entire current road population is sent, not just its first car. This intentionally conservative halo supports the current receiving-space calculations and can later be reduced only after equivalent results are demonstrated. All connectors and incoming lanes relevant to a junction are local to its owner. A network with conflicts spanning different junction IDs is rejected.

## Stage boundaries

| Stage | Immutable inputs | Writer / messages | Required barrier |
|---|---|---|---|
| Start step | End-of-previous-step authoritative cars + refreshed receiving-road ghosts | Owner builds leader index and pre-motion connector / movement context | Ghost exchange must finish first |
| Road motion | Pre-motion snapshot, leader IDs, connector tails, signal/movement availability | Only authoritative owner updates its vehicles | All road motion before connector advance summaries |
| Connector advance | Owner's post-motion state | Original kernel advances connectors; exited vehicles become road-active | Finish before ownership migration |
| Exchange phase 0 | Post-road-motion / post-connector-advance state | Old owner sends migration and receiving-road ghost payloads; new owner becomes sole writer | Apply ownership changes and complete state exchange before requests |
| Summary and requests | Refreshed post-motion state | Each owner summarizes local/ghost state; generates requests only for owned vehicles | Summary before request evaluation |
| Arbitration | Requests sorted by junction, green-before-red pass, stable ID | Junction owner processes requests using original following, conflict and receiving-space checks | All admissions before reservation exchange |
| Reservation exchange | Final reservations, including occupied connectors and new admissions | Upstream junction sends reserved count and movement owner to downstream road owner for every cut lane, including zero | Must finish before pending spawning |
| Spawn | Receiving road's nearest position and authoritative reservation | Road owner inserts the original lowest-ID due pending vehicle if permitted | Finish before phase 1 |
| Exchange phase 1 | Post-spawn states | Refresh receiving-road ghosts; ownership changes here are an error | Finish before next timestep |

The pre-motion snapshot is different from the post-motion snapshot. Reusing stale receiving-space or connector information between them is prohibited. Connector-exit migrations happen before summary/request evaluation so very short downstream roads behave exactly as in the original implementation. Right-on-red still uses the original stop, clearance, arrival and conflict rules.

## Wire format and pools

* State payload: 36 bytes (five mutable integers, four mutable floats).
* State message: 60 bytes (payload + ID, source, destination, new owner, epoch, phase).
* Reservation message: 24 bytes (lane key, source, destination, epoch, reserved count, movement owner).
* Stable IDs, route arrays, departures and network metadata are immutable replicated inputs.
* Each partition has independent current/next vehicle arrays, compact motion state, leader/context/request indexes, summaries, device outbox/inbox and pinned host buffers. State-message capacity defaults to 2N; reservation capacity is 8 times road count. All capacities are allocated before the step loop.
* Overflows abort before state-message application. Messages are ordered deterministically for host routing. Duplicate `(destination, vehicle)` messages, invalid owner/rank/epoch/phase and unexpected migrations are errors. Zero reservation messages explicitly clear previous reservations.
* The prototype broadcasts the full updated ownership map twice per step; the actual H2D bytes are reported separately from the hypothetical compact ownership-control payload. No claimed communication savings may omit this cost.
* GPU sort implementations may allocate scratch internally. Persistent transport/vehicle pools do not imply an allocation-free complete simulator.

## Validation versus production transport

The three logical partitions execute sequentially on one Jetson GPU by default (`0,0,0`), with host-staged message transport. Separate physical devices can be selected (`0,1,2`); this still uses the same conservative, sequential host transport. This is a correctness scaffold, not a concurrent optimized multi-GPU baseline.

The reference has separate buffers and never supplies simulation state to the partitions after initialization. It is used only for comparisons. Every step gathers authoritative output and requires bitwise equality of every 428-byte vehicle record, correct/stale-free ghosts, one authority per vehicle, valid states, road/connector gaps, cross-boundary gaps and conflict exclusion. Verification copies and invariant checks are reported separately from execution time.

The stable-ID address space remains dense: every partition reserves N slots and replicates immutable routes and network metadata. This makes exact priority comparisons straightforward but does **not** deliver distributed memory capacity yet. Distinguish owned, ghost, allocated-slot and message capacities. A subsequent sparse local-ID mapping and peer transport must pass this same oracle before performance claims.

No boundary prediction is allowed to suppress required exchange. A future prior may choose ownership, pool capacities or migration schedules, but cannot change the physical state, skip dependencies, or access held-out future traffic.
