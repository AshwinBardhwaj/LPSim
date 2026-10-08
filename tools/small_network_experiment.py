#!/usr/bin/env python3
"""Reproducible cached-OSM San Pablo / University experiment.

Prepare: python3 tools/small_network_experiment.py prepare
Run: python3 tools/small_network_experiment.py run
Analyze: python3 tools/small_network_experiment.py analyze
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.generate_network import generate_network
ART = ROOT / 'artifacts/san_pablo_university_20261004'
NET = ROOT / 'data/networks'


def write_csv(path, columns, rows):
    with path.open('w') as f:
        w = csv.writer(f); w.writerow(columns); w.writerows(rows)


def prepare():
    import osmnx as ox
    import networkx as nx
    ART.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'cache/ea1901ac2671a60e1651aab79db516ee19d0cb59.json'
    import shutil
    shutil.copy2(source, ART / 'source_cache.json')
    data = json.loads(source.read_text())
    osm = ET.Element('osm', version='0.6')
    for e in data['elements']:
        if e['type'] not in ('node', 'way'): continue
        attrs = {'id': str(e['id']), 'version': '1'}
        if e['type'] == 'node': attrs.update(lat=str(e['lat']), lon=str(e['lon']))
        el = ET.SubElement(osm, e['type'], attrs)
        for n in e.get('nodes', []): ET.SubElement(el, 'nd', ref=str(n))
        for k, v in e.get('tags', {}).items(): ET.SubElement(el, 'tag', k=k, v=v)
    raw = ART / 'source.osm'
    ET.ElementTree(osm).write(raw, encoding='utf-8', xml_declaration=True)
    g = ox.graph_from_xml(raw, simplify=True, retain_all=True)
    # A roughly 600 m square: University, San Pablo, and immediate neighbors.
    selected = [n for n, d in g.nodes(data=True)
                if -122.2955 <= d['x'] <= -122.2890 and 37.8665 <= d['y'] <= 37.8718]
    small = g.subgraph(selected).copy()
    for name, graph in [('san_pablo_university_small', small),
                        ('san_pablo_corridor_small', small.edge_subgraph([
                            (u, v, k) for u, v, k, d in small.edges(keys=True, data=True)
                            if 'San Pablo Avenue' in str(d.get('name', ''))]).copy())]:
        folder = NET / name; folder.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, folder / 'source.graphml')
        generate_network(graph, folder, connectivity='all')
        nodes = list(csv.DictReader((folder / 'nodes.csv').open()))
        edges = list(csv.DictReader((folder / 'edges.csv').open()))
        dg = nx.DiGraph()
        for e in edges: dg.add_edge(int(e['u']), int(e['v']), weight=float(e['length']))
        # Long connected trips exercise several junctions, with no U-turns.
        routes = []
        for u in sorted(dg):
            for v, route in nx.single_source_dijkstra_path(dg, u).items():
                if len(route) < 3: continue
                length = sum(dg[a][b]['weight'] for a, b in zip(route, route[1:]))
                if length >= 250: routes.append((u, v))
        if not routes: raise RuntimeError('No multi-intersection routes')
        write_csv(folder / 'route_pairs.csv', ['origin', 'destination'], routes)
        (folder / 'provenance.json').write_text(json.dumps({
            'source': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'osm_timestamp': data['osm3s']['timestamp_osm_base'], 'attribution': data['osm3s']['copyright'],
            'bbox': [-122.2955, 37.8665, -122.2890, 37.8718],
            'nodes': len(nodes), 'edges': len(edges), 'connectivity': 'all',
            'signal_assumption': 'all junctions, synchronized red [0,20), green [20,40), absolute seconds',
        }, indent=2))
    cases = [('corridor_green', 'san_pablo_corridor_small', 20, 'green'),
             ('corridor_red', 'san_pablo_corridor_small', 20, 'red'),
             ('corridor_cycle', 'san_pablo_corridor_small', 20, 'cycle20')]
    cases += [(f'network_{n}', 'san_pablo_university_small', n, 'cycle20') for n in (20, 100, 400, 1600)]
    cases += [(f'network_{n}_green', 'san_pablo_university_small', n, 'green') for n in (400, 1600)]
    for name, network, count, mode in cases:
        folder = ART / name; (folder / 'data').mkdir(parents=True, exist_ok=True)
        pairs = list(csv.DictReader((NET / network / 'route_pairs.csv').open()))
        rng = random.Random(42)
        trips = sorted((18000 + rng.uniform(0, 180), *map(int, rng.choice(pairs).values())) for _ in range(count))
        if name.startswith('corridor_'):
            # Same-lane platoon on one connected multi-junction corridor route.
            pair = pairs[0]
            trips = [(18000 + 2*i, int(pair['origin']), int(pair['destination'])) for i in range(count)]
        demand = folder / 'demand.csv'
        write_csv(demand, ['dep_time', 'origin', 'destination'], trips)
        # Loader concatenates NETWORK_PATH and demand name; a relative traversal keeps inputs in artifacts.
        od = os.path.relpath(demand, NET / network)
        config = f'''[General]
GUI=false
USE_CPU=false
NETWORK_PATH={NET / network}/
USE_SP_ROUTING=true
USE_PREV_PATHS=false
ADD_RANDOM_PEOPLE=false
LIMIT_NUM_PEOPLE={count}
NUM_PASSES=1
TIME_STEP=0.5
START_HR=5
END_HR=5.1666666667
OD_DEMAND_FILENAME={od}
SHOW_BENCHMARKS=true
REROUTE_INCREMENT=0
NUM_GPUS=1
'''
        (folder / 'data/command_line_options.ini').write_text(config)
        (folder / 'case.json').write_text(json.dumps({'network': network, 'trips': count, 'signal_mode': mode, 'seed': 42, 'duration_seconds': 600}, indent=2))
    print('Prepared', len(cases), 'cases in', ART)


def run(selected=None):
    binary = ROOT / 'build/lpsim'
    for casefile in sorted(ART.glob('*/case.json')):
        folder = casefile.parent; case = json.loads(casefile.read_text())
        if selected and folder.name not in selected: continue
        env = dict(os.environ, LPSIM_SIGNAL_MODE=case['signal_mode'],
                   LPSIM_SIGNALS=str(folder / 'signals.csv'), LPSIM_TRAJECTORIES=str(folder / 'trajectories.csv'), LPSIM_METRICS=str(folder / 'metrics.json'), OMP_NUM_THREADS='2')
        start = time.perf_counter()
        with (folder / 'run.log').open('w') as log:
            result = subprocess.run([str(binary)], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
        (folder / 'execution.json').write_text(json.dumps({'returncode': result.returncode, 'wall_seconds': time.perf_counter()-start, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}, indent=2))
        print(folder.name, result.returncode, flush=True)
        if result.returncode: raise RuntimeError(f'Run failed: {folder}/run.log')
        # Repeated calculation timings with both diagnostic exporters disabled.
        timing_env = dict(env)
        timing_env.pop('LPSIM_SIGNALS'); timing_env.pop('LPSIM_TRAJECTORIES')
        for repeat in range(3):
            timing_env['LPSIM_METRICS'] = str(folder / f'timing_{repeat}.json')
            with (folder / f'timing_{repeat}.log').open('w') as log:
                subprocess.run([str(binary)], cwd=folder, env=timing_env, stdout=log, stderr=subprocess.STDOUT, check=True)
        if folder.name in ('corridor_green', 'network_20', 'network_400', 'network_1600'):
            import shutil
            if shutil.which('ncu') and not os.environ.get('LPSIM_SKIP_PROFILE'):
                timing_env['LPSIM_METRICS'] = str(folder / 'profile_metrics.json')
                command = ['ncu', '--csv', '--page', 'raw', '--metrics',
                           'gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio',
                           '--kernel-name', 'regex:kernel_trafficSimulation', '--launch-skip', '100', '--launch-count', '5', str(binary)]
                with (folder / 'ncu.csv').open('w') as log:
                    profile = subprocess.run(command, cwd=folder, env=timing_env, stdout=log, stderr=subprocess.STDOUT)
                (folder / 'profile_status.json').write_text(json.dumps({'command': command, 'returncode': profile.returncode}))


def analyze():
    summaries = []
    for casefile in sorted(ART.glob('*/case.json')):
        folder = casefile.parent; case = json.loads(casefile.read_text())
        if not (folder / 'metrics.json').exists(): continue
        metrics = json.loads((folder / 'metrics.json').read_text())
        import statistics
        timings = [json.loads(p.read_text())['calculation_seconds'] for p in sorted(folder.glob('timing_*.json'))]
        if timings: metrics['calculation_seconds_no_recording_median'] = statistics.median(timings)
        edge_data = {int(e['uniqueid']): e for e in csv.DictReader((NET / case['network'] / 'edges.csv').open())}
        signal_states = {(float(r['step_start']), int(r['edge_id']), int(r['lane_idx'])): int(r['green']) for r in csv.DictReader((folder / 'signals.csv').open())}
        red_violations = invalid = overlaps = attempts = rejected = 0
        occupied = set()
        previous = {}; stopped = active = changes = crossings = max_queue = 0
        queues = {}; frame = None; max_active = 0; frame_active = 0; timeline = {}
        for r in csv.DictReader((folder / 'trajectories.csv').open()):
            t = float(r['timestamp']); vehicle = int(r['vehicle_id']); edge = int(r['edge_id']); lane = int(r['lane_idx']); pos = float(r['pos_m']); speed = float(r['speed'])
            if frame != t:
                if frame is not None: timeline[frame] = (frame_active, sum(queues.values()), max(queues.values(), default=0))
                max_queue = max(max_queue, max(queues.values(), default=0)); max_active = max(max_active, frame_active)
                queues = {}; occupied = set(); frame_active = 0; frame = t
            frame_active += 1; active += 1
            attempts += int(r['lane_change_attempted']); rejected += int(r['lane_change_gap_rejected'])
            cell = (edge, lane, int(pos))
            overlaps += cell in occupied
            occupied.add(cell)
            # CSV edge lengths and CUDA float32 positions differ by a few micrometers.
            # A 0.1 mm upper-bound tolerance avoids classifying a rounded endpoint as overshoot.
            invalid += not math.isfinite(pos) or not math.isfinite(speed) or lane < 0 or lane >= int(edge_data[edge]['lanes']) or pos < 0 or pos > float(edge_data[edge]['length']) + 1e-4 or speed < 0
            if speed < 0.5:
                stopped += 1; queues[edge, lane] = queues.get((edge, lane), 0) + 1
            old = previous.get(vehicle)
            if old and t-old[0] <= .501:
                if edge == old[1] and lane != old[2]: changes += 1
                stop = max(0, float(edge_data[old[1]]['length']))
                green = signal_states[(t-.5, old[1], old[2])]
                if not green and old[3] <= stop and (edge != old[1] or pos > stop + .001): red_violations += 1
                # Edge transitions on red can legally clear vehicles already beyond stop line.
                if edge != old[1] and (case['signal_mode'] == 'red' or (case['signal_mode'] == 'cycle20' and int((t-.5)//20)%2 == 0)): crossings += 1
            previous[vehicle] = (t, edge, lane, pos)
        if frame is not None: timeline[frame] = (frame_active, sum(queues.values()), max(queues.values(), default=0))
        write_csv(folder / 'timeseries.csv', ['timestamp', 'active', 'stopped', 'max_stopped_per_lane'],
                  [(18000 + .5*i, *timeline.get(18000 + .5*i, (0, 0, 0))) for i in range(1, 1201)])
        max_queue = max(max_queue, max(queues.values(), default=0))
        summary = dict(case=folder.name, **case, **metrics, active_vehicle_timesteps=active,
                       stopped_fraction=stopped/max(active, 1), max_stopped_per_lane=max_queue,
                       lane_changes=changes, lane_change_attempts=attempts, lane_change_gap_rejections=rejected, transitions_during_red=crossings, red_stopline_violations=red_violations, invalid_samples=invalid, occupied_cell_duplicates=overlaps,
                       note='CUDA stop line is edge end (clearance=0). Duplicate occupancy flags a model failure, not validated collision freedom.')
        (folder / 'analysis.json').write_text(json.dumps(summary, indent=2)); summaries.append(summary)
    (ART / 'summary.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--case', action='append', help='Run only a named prepared case (repeatable)')
    parser.add_argument('--output', type=Path, help='Alternate artifact directory')
    args = parser.parse_args()
    if args.output: ART = args.output.resolve()
    if args.action == 'run': run(args.case)
    else: globals()[args.action]()
