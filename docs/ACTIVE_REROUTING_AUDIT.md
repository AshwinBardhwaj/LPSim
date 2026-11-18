# Mid-simulation road closures and rerouting audit

The current source does not provide a safe active-vehicle route replacement API.
It can compute routes in departure batches; that is not the requested event-driven
rerouting of vehicles already on the road. Implementing it is feasible, but more
involved than adding vehicle parameters or a trajectory inspector.

## Evidence in this checkout

- `src/routing/shortest_path.cpp`, `RoutingWrapper`: filters OD pairs by departure
  time, then builds a contraction-hierarchy routing graph for that batch.
- `convertPathsToCUDAFormat`: checks that an already assigned route start offset
  is not reassigned. The packed route array is assumed to preserve prior offsets.
- `src/simulator/traffic_simulator.cpp`, `simulateInGPU`: the inner timestep loop
  has no scheduled event queue. Route updates happen outside it at batch boundaries.
- `src/simulator/cuda_traffic_sim.cu`, `b18updateStructuresCUDA_n`: reallocates route
  and edge buffers, rebuilds GPU vehicle vectors using origin partitions, and adds
  initial ghost copies. Calling it on an arbitrary active snapshot risks putting
  vehicles on the wrong GPU and invalidating state/index assumptions.
- `b18SimulateTrafficCUDA`: synchronizes motion kernels, then processes migration
  and ghost-lane updates. An event boundary must be after that entire operation,
  not merely after the first kernel synchronization.

## SUMO reference

[SUMO routing](https://sumo.dlr.de/docs/Simulation/Routing.html) supports
`rerouteTraveltime`, or `findRoute` followed by `setRoute`.
[TraCI vehicle changes](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html)
require the replacement route to start on the vehicle's current edge. SUMO also
supports temporary access closures through rerouters. This is a useful contract
for LPSim, not evidence that LPSim currently implements the same functionality.
Sources checked October 23, 2026.

## Smallest practical implementation plan

1. Add a sorted event input (simulation seconds, close/reopen, physical edge ID).
   Apply each event once at the first completed timestep at or after its time;
   log scheduled and applied timestamps. Initially allow vehicles already on a
   closed edge to clear it; block new entrants. Distinguish access restrictions
   from a physical blockage that also stops occupants.
2. At that boundary, synchronize all devices and finish migrations. Collect one
   authoritative state per vehicle ID, retaining the actual owner GPU. Preserve
   position, speed, lane, class parameters, accumulated metrics, and active status.
3. Mark affected remaining routes, including not-yet-departed vehicles. Compute
   suffixes from the end of the current edge to the original destination using a
   CPU Dijkstra search with closed edges excluded and declared travel-time weights.
   Reusing the existing contraction hierarchy requires rebuilding it after changes.
   A small event-only Dijkstra implementation avoids changing steady-state kernels.
4. Preserve the current edge as the first edge of each replacement. Defer a turn
   change while a vehicle is committed inside an intersection. If no route exists,
   hold the vehicle safely upstream and retry on reopen; never delete or teleport it.
5. Stage new packed paths and validate topology, bounds, destinations and memory
   capacity before publishing. Patch route offsets and next-edge/turn lane caches
   in place on the owning GPUs and their ghosts. Keep physical position and current
   lane occupancy. Rebuild any derived leader/turn indices after publication.
6. Record a route version and event log with each change. Export physical edge IDs
   and versioned route arrays so visualization and route-distance metrics do not
   compare a changed route to the original static route file.

Start with one GPU and a small diamond network: close the future edge mid-run,
verify a detour, then reopen it. Cover no-path, vehicles already on the closed
edge, a vehicle inside a junction, simultaneous events, skipped event times,
unaffected vehicles, duplicate ghost copies, and allocation failure. Assert
vehicle count/identity conservation and no instantaneous position/speed changes.
Then validate a partition-boundary detour on multiple GPUs. Static partitioning
can remain in place: existing migration handles ownership as vehicles move.
Predictive allocation and state prediction profiling are outside this work.

This change set contains the investigation and plan; it does not enable road
closure events or claim safe active rerouting is already implemented.
