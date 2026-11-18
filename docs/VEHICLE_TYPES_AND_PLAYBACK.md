# Vehicle parameters and individual trajectory playback

## Per-vehicle profiles

Rebuild with `cmake --build build -j2`. The vehicle struct changed, so rebuild any
standalone CUDA experiments that embed it; do not reuse binary struct snapshots
from the previous layout.

Set `LPSIM_VEHICLE_PROFILES=/absolute/path/vehicle_profiles.csv` when running the
simulator. Use the exact CSV header below (SI units):

```csv
vehicle_id,type_id,min_gap_m,time_headway_s,max_speed_mps,accel_mps2,decel_mps2
0,1,2,1.5,12,1.5,2.5
1,2,5,2.5,6,0.8,1.5
```

IDs refer to simulator vehicle IDs, not row numbers in trajectories. Unlisted
vehicles retain global defaults. Unknown/duplicate IDs, non-finite numbers,
nonpositive time headway/speed/acceleration/braking, and negative gaps fail before
GPU upload. Type IDs are user-defined categories (0–65535), exported as
`vehicle_type`; the numbers above are illustrative, not calibrated sedan/truck data.

The same IDM formulation uses each vehicle's acceleration `a`, comfortable braking
`b`, time headway `T`, minimum gap, and desired speed capped by the current road's
speed limit. IDM braking is not hard-clamped to `b`; collision/signal safeguards
can stop a vehicle faster. Maximum speed is the IDM free-flow target, not an
external hard velocity clamp. Minimum gap also affects insertion and lane-change
checks. It is not a physical vehicle body length. Time headway (seconds) and
standstill gap (meters) are different quantities; neither represents road grade.

The previously present `idm_config.h` presets were not wired into this execution
path. This implementation provides explicit assignments rather than guessing a
fleet mix or treating every truck as the same vehicle.

One-command deterministic GPU demonstration (two departures on the same corridor):

```bash
python3 tools/run_vehicle_type_demo.py --output artifacts/vehicle-types-demo
```

Use a new output directory. It uses the already-built simulator, writes isolated
configuration/demand/profile inputs, validates distinct speed behavior, and creates
`playback/index.html`, coordinate CSV, and input/binary hash manifests. It does not
run a profiler or change your main simulation config.

## Inspect recorded vehicles

```bash
python3 tools/export_vehicle_playback.py \
  --network data/networks/berkeley \
  --routes 0_route5to12.csv \
  --trajectories trajectories_physical.csv \
  --vehicles 0 1 \
  --output reports/vehicle-comparison
```

Open the output `index.html` directly: it has no CDN, server, or map-tile dependency.
It shows network geometry, animated markers, vehicle selection, optional comparison
traces, absolute or first-observation-aligned time, recorded speed, derived
acceleration, and geographic coordinates. Two vehicle IDs can represent different
routes or departure times within the same recording; it does not simulate a
counterfactual or import separate runs into one comparison.

`observations.csv` contains only observed frames. Acceleration is backward
`(speed_now-speed_before)/(time_now-time_before)`, with nulls at continuity breaks.
Between-frame geometry interpolation is only for animation. Legacy recordings
without speed show unknown speed/acceleration rather than invented values. Exact
duplicates are removed and conflicting timestamps break continuity. The manifest
records input/source/output SHA-256 hashes and validation counts. Coordinates are
centerline longitude/latitude, without elevation; fuel/energy models require
additional physical parameters and grade data. Travel durations reflect recorded
coverage, not necessarily the full insertion-to-arrival trip.

See [the rerouting audit](ACTIVE_REROUTING_AUDIT.md) for the second requested task.

## Validation

```bash
python3 -m unittest discover -s tests -p 'test_vehicle_playback.py'
g++ -std=c++17 -Iinclude -Isrc/simulator tests/test_vehicle_profiles.cpp -o /tmp/lpsim-profile-test
/tmp/lpsim-profile-test /tmp/lpsim-profile-test.csv
node tests/test_vehicle_playback.cjs reports/vehicle-comparison/index.html
```

The Node check exercises rendering calculations and controls with a minimal DOM;
it is not a screenshot or full browser integration test. On this development run,
CUDA compilation and the host/data tests passed. The GPU smoke test could not
initialize CUDA inside the sandbox, and automatic approval review failed when
requesting device access. Its script is provided for validation on the Jetson;
live heterogeneous GPU behavior has not yet been verified by this change set.
