# Small San Pablo–University experiment

Run `python3 tools/small_network_experiment.py prepare` to recreate the two OSM
networks in `data/networks` from the retained Berkeley Overpass cache. The script
uses `tools/generate_network.py` for CSV normalization and saves source GraphML.

Build with `cmake --build build -j2`, then run:

```sh
python3 tools/small_network_experiment.py run
python3 tools/small_network_experiment.py analyze
python3 tools/report_small_network.py
```

Results and the detailed report are in
`artifacts/san_pablo_university_20261004/REPORT.md`. `run --case corridor_red`
selects one prepared case. Re-running overwrites that case's outputs.

The runner launches the binary from each artifact case directory, containing its
own `data/command_line_options.ini`. It does not change the root configuration.

CUDA diagnostic environment variables:

- `LPSIM_SIGNAL_MODE=cycle20`: synchronized 20 s red / 20 s green at all ordinary
  junctions, absolute simulation seconds. `red` and `green` hold a fixed state.
  Unset retains the original approach-cycling scheduler. Red-light braking now
  applies to ordinary ground vehicles in either scheduler. Vertiport control is
  unchanged. This is synthetic control, with no yellow or conflict resolution.
- `LPSIM_SIGNALS=/absolute/output.csv`: actual per-step device signal states by
  physical edge and lane. `step_start` is the time at which the state is applied.
- `LPSIM_TRAJECTORIES=/absolute/output.csv`: post-step observations, including
  lane-change attempts and gap-rejection flags. Those flags record a gap decision,
  not a guarantee of a successful maneuver or collision-free lane occupancy.
- `LPSIM_METRICS=/absolute/output.json`: synchronized stepping time, recording
  time, steps, completed/active/pending trips. This is host-observed calculation
  time, not pure kernel time. Exporter overhead is measured separately.

The experiment preserves car following and existing lane-change decisions.
Validation deliberately reports failed occupancy checks: dense traffic must not
be called validated merely because signals and completion accounting pass.
Nsight Compute counter access currently requires privileges unavailable to the
runner; profiler logs retain the failure instead of reporting fabricated metrics.


## Occupancy regression

The reservation/rollback repair and before/after results are documented in
`artifacts/occupancy_fix_20261004/REPORT.md`. Repeat the prepared cases without
overwriting the earlier baseline:

```sh
LPSIM_SKIP_PROFILE=1 python3 tools/small_network_experiment.py run --output artifacts/occupancy_fix_20261004
python3 tools/small_network_experiment.py analyze --output artifacts/occupancy_fix_20261004
python3 tools/validate_occupancy.py artifacts/occupancy_fix_20261004
nvcc -arch=sm_87 -std=c++17 tests/test_lane_occupancy.cu -o /tmp/test_lane_occupancy
/tmp/test_lane_occupancy
```

Change the CUDA architecture for other hardware. The helper test stresses
neighboring-byte contention and protects old occupied cells from other vehicles
until the next timestep. The traffic validator checks recorded cell uniqueness,
finite/valid vehicle states, red compliance, phases, duration and accounting.
Position bounds allow 0.1 mm for CSV versus float32 endpoint rounding.
