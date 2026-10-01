# Recorded vehicle animation

From the repository root, rebuild and record a fresh run:

```bash
cmake --build build -j2
LPSIM_TRAJECTORIES=trajectories_physical.csv ./build/lpsim
python viz/process_routes.py
python -m http.server 8080 --directory viz
```

Open http://localhost:8080/animated_sim/ . The viewer reads `viz/trips.json`;
its old local `animated_sim/trips.json` is no longer used. At 1x, one real second
advances one simulation second. Red dots indicate stopped/slow vehicles (<2 m/s);
short trails and persistent dots make queues visible. Color speed is estimated
between geographic samples; authoritative speeds remain in the CSV.

The default converter route file is `0_route5to12.csv`. Use `--routes` for other
simulation hours. Use the routes from the **same run** as the trajectories.
The route file's final field is distance, not departure time.

`LPSIM_TRAJECTORIES` enables recording every simulation step and specifies the
output file (overwritten each run). Multiple simulation passes append `.passN`
to that filename. Recording adds GPU readback and disk costs; unset the variable
for normal benchmarks. Vehicle state is gathered into a compact GPU snapshot
after simulation and migration complete, without modifying the host vehicle
array. Timestamps label the end of each step (`currentTime + deltaTime`). Only
active vehicles are emitted; no departure-to-arrival extrapolation or synthetic
terminal samples are added.

Each row contains `vehicle_id,timestamp,edge_id,route_index,lane_idx,pos_m,speed,edge_id_kind`.
The route pointer indexes the global CUDA route array; its lane-map entry is
translated through the inverse physical-edge mapping, never treated as an OSM ID.
`edge_id_kind=uniqueid` explicitly distinguishes the new format from old dumps.
Ghost duplicates are handled by the converter: identical records are deduplicated;
conflicting same-time states split the path and are excluded.

The converter sorts observations, retains stops, validates the route index and
physical edge together, and uses the network's simulation lengths. It interpolates
along oriented WKT LineStrings when supplied. The present Berkeley network has
no geometry column, so its rendering uses straight node-to-node segments. Within
each sample interval, additional polyline vertices prevent corner-cutting; their
times use linear interpolation within that interval, not trip-average speed.

Missing routes, invalid positions, reversed movement, disconnected edges, and
route-distance speeds above `--max-speed` (default 45 m/s) are diagnosed in
`trips.diagnostics.json`. Invalid samples break paths. Route overflow discards the
remaining vehicle samples. Conflicting observations are never resolved by guessing.
Legacy ambiguous dumps are rejected, preserving any existing JSON. Actual
completed trip timing is not extrapolated beyond the last active sample.

Validation: six Python regression tests plus a native Jetson three-minute run
(158 departures, 26,517 recorded samples, 156 drawable trips). The initial preview used that short run. The current full-run export contains
12,800,594 accepted samples and 20,000 trips, spanning the seven-hour simulation.
No multi-GPU hardware validation has been performed.

```bash
python -m unittest discover -s tests -p test_process_routes.py
```

## Full-run browser playback

The converter now also writes `viz/playback/manifest.json` and five-minute JSON
windows. Every original sample is retained, with boundary observations and three
seconds of overlap for continuous trails. The browser loads one window at a time
and pauses simulation time while loading. This avoids parsing the complete
616 MB seven-hour export in the browser. `trips.json` remains the full export.

For an already converted run, no simulation rerun is needed:

```bash
python viz/prepare_playback.py
python -m http.server 8080 --directory viz
```

Open `/animated_sim/` and refresh once after updating the viewer. Each preparation
creates a new playback directory and replaces the manifest only after completion.
Older playback directories are retained so already open viewers continue working.
A server `BrokenPipeError` means the browser closed a download connection; it
does not mean the simulation failed.

## Lane-specific review

Build an isolated, portable review package from the existing run:

```bash
python tools/package_lane_review.py --output artifacts/berkeley-lane-review
python -m http.server 8080 --directory artifacts/berkeley-lane-review
```

Choose a new output name if the directory already exists. Use `/animated_sim/`.
The package contains `rebuild.py`, captured inputs/code and a SHA-256 manifest.
The full CSV is sorted on disk to bound memory on the Jetson. Lane positions
use a configurable schematic width (default 3.2 m), with lane 0 leftmost in travel
direction. Recorded samples remain authoritative; intermediate points and
lateral movement between timestamps are interpolation. Hover a vehicle to see
recorded lane/edge values; pause and scrub to inspect a maneuver. Rendering
cannot establish whether a recorded lane-change decision was physically safe.

Signal findings and the proposed localized repair are in `docs/GPU_SIGNAL_AUDIT.md`.
No signal dynamics have been changed in this review package.
