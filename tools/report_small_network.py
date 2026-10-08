#!/usr/bin/env python3
"""Validate recorded signal contracts and render the small-network results."""
import csv
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lpsim-matplotlib')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.small_network_experiment import ART, NET
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

summaries = json.loads((ART / 'summary.json').read_text())
checks = []
for case in summaries:
    folder = ART / case['case']
    errors = 0
    for r in csv.DictReader((folder / 'signals.csv').open()):
        t = float(r['step_start'])
        expected = case['signal_mode'] == 'green' or (case['signal_mode'] == 'cycle20' and int(t // 20) % 2 == 1)
        errors += int(r['green']) != int(expected)
    checks.append({'case': case['case'], 'signal_state_mismatches': errors,
                   'fixed_1200_steps': case['steps'] == 1200,
                   'trip_accounting': case['completed'] + case['active'] + case['pending'] == case['trips'],
                   'red_compliance': case['red_stopline_violations'] == 0,
                   'valid_samples': case['invalid_samples'] == 0,
                   'distinct_occupied_cells': case['occupied_cell_duplicates'] == 0})
(ART / 'validation.json').write_text(json.dumps(checks, indent=2))
assert all(c['signal_state_mismatches'] == 0 and c['fixed_1200_steps'] and c['trip_accounting'] and c['red_compliance'] for c in checks)
by_name = {s['case']: s for s in summaries}
assert by_name['corridor_red']['completed'] == 0
assert by_name['corridor_green']['completed'] == by_name['corridor_cycle']['completed'] == 20
assert by_name['corridor_red']['max_stopped_per_lane'] > 1

# Geographic view uses saved OSM geometry, not invented straight street shapes.
import osmnx as ox
fig, ax = plt.subplots(figsize=(8, 8))
g = ox.load_graphml(NET / 'san_pablo_university_small/source.graphml')
labels = set()
for u, v, d in g.edges(data=True):
    geom = d.get('geometry')
    x, y = geom.xy if geom is not None else ([g.nodes[u]['x'], g.nodes[v]['x']], [g.nodes[u]['y'], g.nodes[v]['y']])
    name = str(d.get('name', 'unnamed connector'))
    main = name in ('San Pablo Avenue', 'University Avenue')
    ax.plot(x, y, color='#cf553c' if main else '#8793a1', linewidth=2 if main else 1)
    if name not in labels and name != 'unnamed connector':
        ax.text(x[len(x)//2], y[len(y)//2], name, fontsize=7)
        labels.add(name)
ax.scatter([d['x'] for _,d in g.nodes(data=True)], [d['y'] for _,d in g.nodes(data=True)], s=10, color='#23364b')
ax.set_aspect(1 / .789); ax.set_title('San Pablo–University: 41 OSM nodes, 91 directed links')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.ticklabel_format(useOffset=False)
fig.text(.12, .02, '© OpenStreetMap contributors • cached 2026-09-17 • synthetic signals at every node', fontsize=8)
fig.savefig(ART / 'network.png', dpi=160, bbox_inches='tight'); plt.close(fig)

cycle = sorted((s for s in summaries if s['case'].startswith('network_') and s['signal_mode']=='cycle20'), key=lambda s:s['trips'])
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, key, title in zip(axes, ['calculation_seconds_no_recording_median','stopped_fraction','completed'], ['Calculation time (s), median of 3','Fraction of active samples stopped','Completed trips in 600 s']):
    ax.plot([s['trips'] for s in cycle], [s[key] for s in cycle], 'o-')
    ax.set_xscale('log'); ax.set_xlabel('Requested trips'); ax.set_title(title); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(ART / 'results.png', dpi=160); plt.close(fig)

rows = ['| Case | Completed / requested | Active / pending at end | Stopped samples | Max stopped / lane | Gap rejections | Duplicate cells | Calculation ms¹ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
for s in summaries:
    rows.append(f"| {s['case']} | {s['completed']} / {s['trips']} | {s['active']} / {s['pending']} | {s['stopped_fraction']:.1%} | {s['max_stopped_per_lane']} | {s['lane_change_gap_rejections']} | {s['occupied_cell_duplicates']} | {1000*s['calculation_seconds_no_recording_median']:.1f} |")
report = '''# San Pablo–University small-network experiment

Implemented and executed on NVIDIA Orin, CUDA architecture 87. Both networks are
under `data/networks`; all inputs, run configurations, recordings and results are
under this artifact directory. The cached OSM snapshot is dated 2026-09-17.

![Network](network.png)

## Experiment definition

- Corridor: 13 nodes / 11 directed links, the two directed San Pablo carriageways;
  disconnected boundary pieces retained, demand restricted to connected paths.
- Intersection neighborhood: roughly 570 × 590 m, 41 nodes / 91 directed links;
  San Pablo, University, and immediate neighboring streets. OSM geometry retained
  in GraphML; simulator uses directed link lengths and lane counts.
- 600 seconds from 05:00, timestep 0.5 seconds, one GPU, fixed demand seed 42.
  Traffic is injected during the first 180 seconds; the corridor platoon uses
  20 cars at two-second departure intervals on one route. Entry lane selection
  remains the simulator's existing behavior, so cars can occupy multiple lanes.
- Every node has synthetic synchronized signals: red [0,20), green [20,40),
  repeating. The CUDA stop line is the link end (its clearance constant is zero).
  This is a stop-and-go experiment, not a conflict-free movement signal plan.
- `signals.csv` records actual device signal states used at `step_start`;
  `trajectories.csv` records state after that step. Car following remains enabled.
- Low, intermediate, and high demand are candidate congestion regimes, classified
  using observed stopping, queues and completion rather than labels alone.

## Results

'''+ '\n'.join(rows) + '''

¹ Median of three runs with trajectory/signal recording disabled. Calculation
includes synchronized GPU stepping, lane-map clearing and migration orchestration;
it excludes routing, initialization and file output. It is not pure kernel time.
Recorded runs also report calculation and recording time separately. These short
runs are an instrumentation baseline, not a reliable complexity-scaling conclusion.

![Results](results.png)

## Sanity checks and limitations

The recorded signal states match the requested phases at every sampled step.
All cases execute exactly 1,200 steps with consistent final trip accounting.
No sampled edge transition crosses a red signal. Forced red holds the platoon
and produces multiple stopped cars per lane; cycling green releases it and all
20 corridor trips finish. The low-demand neighborhood also clears all trips.

**Congested traffic is not fully validated.** `validation.json` exposes duplicate
one-meter occupied cells and any invalid trajectory states, rather than treating
successful program termination as correct traffic behavior. Duplicate counts are
vehicle-samples, not unique crashes. Stopped-per-lane is an aggregate proxy, not
a contiguous physical queue length. Dense cases need occupancy/collision and
rollback investigation before partition performance claims are meaningful.

Lane-change attempts and rejected gaps are exported explicitly. The code already
contains a discretionary speed/deceleration trigger as well as turn-related logic.
Attempts require speed above 3 m/s, so a fully stopped car does not retry a lane
change. The existing edge-transition logic does not require a successful mandatory
lane change before crossing. Consequently this experiment cannot establish the
claim that a blocked turning vehicle must stall all followers: that behavior is
not enforced by the current model. Queues behind stopped vehicles are observed,
but attributing them specifically to a failed maneuver requires a controlled
turn-lane fixture and stronger movement constraints.

Nsight Compute captures were attempted (`ncu.csv`, `profile_status.json`). This
host refuses counter access without root privileges; occupancy and warp-efficiency
metrics are therefore unavailable, not zero. No profiler permission settings were
changed. The Python network-generator test suite could not run because pytest is
not installed; the actual exporter was exercised on both cached-OSM networks.

## Reproduction

From the repository root:

```sh
python3 tools/small_network_experiment.py prepare
cmake --build build -j2
python3 tools/small_network_experiment.py run
python3 tools/small_network_experiment.py analyze
python3 tools/report_small_network.py
```

Use `run --case corridor_red` to repeat one case. GPU device access is required.
Preparation uses the retained OSM cache and the normalization/export functions in
`tools/generate_network.py`; `source.graphml` can also be passed directly to that
exporter's `--graphml` option. The root simulator config is not modified.
`source_snapshot.tar.gz`, `lpsim`, build logs, manifest hashes, per-case configs,
demand CSVs and execution metadata preserve the implementation and inputs used.
Fixed inputs do not imply bitwise deterministic GPU scheduling at high density.

Next corrective work: repair and regression-test same-lane occupancy/rollback,
then add a constrained blocked-turning fixture; only then accept congested results
for static partition comparisons.
'''
(ART / 'REPORT.md').write_text(report)
print('Signal, duration, accounting and corridor queue/release assertions passed.')
print('See validation.json for traffic-model failures and REPORT.md for conclusions.')
