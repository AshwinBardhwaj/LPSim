# Automatic traffic and computational reports

Generate an offline report from the captured Berkeley lane-review package:

```bash
python tools/generate_run_report.py --run-dir artifacts/berkeley-lane-review --output reports/berkeley-next
```

Choose a new output directory on each invocation. Requires Python 3.10+, NumPy,
and Matplotlib. No web server is needed: open report.html or report.pdf directly.
PNG figures are suitable for slides. CSVs contain every observed vehicle and road
segment, and five-minute network summaries. summary.json contains totals and
quality counts; manifest.json records source SHA-256 hashes, software versions,
arguments and output hashes. Generator and geometry-reader source are captured.
Keep the original input package with the report to regenerate it.

## Definitions and coverage

- VMT = sum of valid observed route-distance increments / 1609.344.
- VHT = sum of valid observed inter-sample vehicle seconds / 3600.
- The per-vehicle observed span is last minus first recorded timestamp. Valid
  observed time excludes rejected gaps. Neither is a complete trip travel time:
  this exporter starts after insertion and stops before a terminal observation.
- Segment speed = accumulated distance divided by accumulated vehicle time.
  Vehicle sampled-speed mean is a separate column; these measures need not agree.
- Segment length / segment mean speed is a **travel-time proxy**, not the mean of
  measured complete traversals. Entry/exit events are needed for exact travel times.
- Across sampled edge transitions, distance is known from route position, but
  boundary crossing times are estimated proportionally to traveled distance.
  Time is apportioned to roads and five-minute bins using this assumption.
- Stopped intervals have reconstructed speed below 0.1 m/s. Delay is estimated
  against speed-limit travel time, not calibrated free-flow or signal-only delay.
- Invalid samples/conflicts break continuity. Backward, disconnected, or >45 m/s
  intervals are excluded and reported. VMT/VHT represent recorded coverage;
  unknown or missing measurements are not filled with zero.

## Record computational measurements on the next run

Run from an isolated simulation working directory with its own `data/command_line_options.ini`
and output files, or intentionally use your usual run directory. The recorder does
not alter the simulator's working directory outputs or redirect its trajectories.

```bash
python /path/to/LPSim/tools/record_run_metrics.py --output capture --cwd /path/to/run -- /path/to/LPSim/build/lpsim
```

The recorder saves run.log, capture.json, raw timestamped tegrastats.jsonl and
telemetry.csv. tegrastats must be installed and permitted to read the device.
Inspect telemetry-errors.log if unavailable. Whole-command runtime includes routing,
setup, simulation and I/O; it is not kernel time. GR3D is GPU activation percentage;
RAM is shared system memory; VDD_IN/POM_5V_IN is module input power, not GPU-only power.
Use a new directory for each capture. No sudo or global tegrastats termination is used.

Then supply the CSV to the report for the **same run**:

```bash
python tools/generate_run_report.py --run-dir artifacts/NEW-RUN --output reports/NEW-RUN --telemetry capture/telemetry.csv
```

Optional measured inputs (blank cells mean unavailable):

| Option | Required time axis | Other columns |
| --- | --- | --- |
| --telemetry | wall_seconds | gpu_util_pct, ram_used_mb, temperature_c, power_w |
| --steps | wall_seconds | step_ms, simulation_seconds |
| --partitions | simulation_seconds | gpu_id, active_vehicles, migrations, transfer_bytes |
| --warps | wall_seconds | gpu_id, achieved_occupancy_pct, active_warps, eligible_warps |

These files must be exported by matching instrumentation/profiling. The generator
accepts them; the current trajectory exporter does **not** collect step timings,
partition migrations or warp counters. The report explicitly marks them missing.
Single-GPU telemetry cannot establish multi-GPU scaling or partition balance.

## Targeted warp/kernel profiling

`tools/profile_sim.sh` now targets build/lpsim and exports an Nsight Compute raw CSV.
The profiler runs the simulator and writes its usual outputs, so use a separate
working directory/config for profile runs. Set LPSIM_BINARY to an absolute binary
path when invoking the script from another directory.

```bash
LPSIM_BINARY=/path/to/LPSim/build/lpsim LPSIM_PROFILE_SKIP=100 /path/to/LPSim/tools/profile_sim.sh ncu profile
python tools/generate_run_report.py --run-dir artifacts/PROFILE-RUN --output reports/PROFILE-RUN --ncu profile_kernels.csv
```

Profiling five selected launches can perturb execution. Nsight Compute metrics
are plotted by profiled launch sequence, **not wall-clock time**. They do not
constitute continuous warp utilization. Use --warps only with a profiler export
that actually contains measurement timestamps. Availability of hardware counters
and continuous GPU metric sampling depends on the installed tools/device permissions.
Nsight Systems is optional and is not currently on PATH on this Jetson.

References: NVIDIA's [tegrastats documentation](https://docs.nvidia.com/jetson/archives/r36.4.4/DeveloperGuide/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
and [Nsight Compute CLI documentation](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html).

## Generate automatically after a successful command

For a command that produces the run package (`inputs/network/{nodes,edges}.csv`,
`inputs/routes.csv`, and `inputs/trajectories.csv`), use:

```bash
python tools/record_run_metrics.py --output capture/NEW-RUN --report-run-dir artifacts/NEW-RUN -- COMMAND ARGS
```

After the command succeeds, the recorder generates `capture/NEW-RUN/report/report.html`
and `report.pdf`, figures, and CSVs. The command must produce the package for that
same run; the recorder does not package raw simulator outputs. Failed commands
retain their capture without generating a success report. Report failures return a
nonzero status and preserve the captured data.

To generate separately, use `--capture capture/NEW-RUN` to discover `telemetry.csv`,
`steps.csv`, `partitions.csv`, and `warps.csv`. Explicit CSV options override discovery.
Empty telemetry is marked unavailable. `--bin-seconds 60` selects one-minute traffic
summaries. Road figures show speed, VMT, VHT, and estimated travel time; network
figures show both VMT and VHT over time. Partition data may use `partition_id` to
separate partitions independently of `gpu_id`.

The report additionally exports `segment_time_bins.csv` for per-road metrics over
simulation time. `mean_estimated_traversal_s` averages visits with both entry and
exit boundary crossings covered by the recording; crossing times are interpolated,
and first/last censored visits are excluded. It is not an exact event measurement.
Bins with less than 95% temporal coverage are omitted from the traffic timeline
figure to avoid presenting the final half-second as a full five-minute traffic drop;
all bins and their coverage_seconds remain in the CSVs.

To revise figures without reprocessing the complete trajectory CSV, supply
`--from-tables reports/PREVIOUS` and a new output directory. The generator checks
input and cached-table hashes and requires an identical bin width before reuse.

## Formatted full-run PDF and browser reader

```bash
python tools/format_run_report.py --tables reports/berkeley-performance-final --output reports/berkeley-full-formatted
```

Use a new output directory. This formatter verifies the recorded inputs and source
CSV checksums, then produces a 12-page landscape PDF with summary percentiles,
traffic figures, road maps, metric definitions, and provenance. It currently targets
the Berkeley historical recording without compute telemetry; its compute coverage
page explicitly lists unavailable measurements. It does not collect new metrics.

`read-report.html` embeds every report page and the complete PDF download, so it
can be opened independently in a browser without PDF preview support. The adjacent
`LPSim-full-report.pdf` can also be opened directly in a standard PDF reader.
When the workspace is on a remote machine, download/copy the file to your computer
first; a remote filesystem path is not a local download.
