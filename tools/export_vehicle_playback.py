#!/usr/bin/env python3
"""Build a portable individual-vehicle playback and observation CSV from one run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from viz.process_routes import convert, load_edges


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def export(network, routes, trajectories, ids, output):
    if output.exists():
        raise ValueError('Choose a new output directory')
    with tempfile.TemporaryDirectory() as temp:
        subset = Path(temp) / 'selected.csv'
        found = set()
        with trajectories.open() as source, subset.open('w') as target:
            reader = csv.DictReader(source)
            writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row['vehicle_id'] in ids:
                    writer.writerow(row)
                    found.add(row['vehicle_id'])
        if found != ids:
            raise ValueError(f'No observations for vehicle IDs: {sorted(ids - found)}')
        trips, diagnostics = convert(network, routes, subset)
    if not trips:
        raise ValueError('No valid continuous trajectories for selected vehicles')
    output.mkdir(parents=True)
    columns = ['vehicle_id', 'segment', 'timestamp_s', 'longitude', 'latitude',
               'edge', 'route_index', 'lane', 'pos_m', 'speed_mps', 'acceleration_mps2']
    with (output / 'observations.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for segment, trip in enumerate(trips):
            for point, sample in zip(trip['path'], trip['samples']):
                if sample is not None:
                    writer.writerow(dict(vehicle_id=trip['vendor'], segment=segment,
                                         timestamp_s=point[2], longitude=point[0], latitude=point[1], **sample))
    edges = load_edges(network)
    payload = dict(trips=trips, roads=[e.points for e in edges.values()], diagnostics=diagnostics)
    (output / 'trajectories.json').write_text(json.dumps(payload, allow_nan=False))
    template = ROOT / 'viz/vehicle_playback.html'
    html = template.read_text().replace('/*TRAJECTORY_DATA*/null', json.dumps(payload, allow_nan=False).replace('<', '\\u003c'))
    (output / 'index.html').write_text(html)
    shutil.copy2(Path(__file__), output / 'export_vehicle_playback.py')
    inputs = [network / 'nodes.csv', network / 'edges.csv', routes, trajectories]
    manifest = dict(vehicle_ids=sorted(ids), inputs={str(p.resolve()): digest(p) for p in inputs},
                    source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                                   [Path(__file__), ROOT / 'viz/process_routes.py', template]},
                    diagnostics=diagnostics,
                    definitions={'speed': 'Recorded instantaneous speed, m/s; null if unavailable.',
                                 'acceleration': 'Backward speed difference / elapsed seconds; null at segment start or if speed unavailable.',
                                 'coordinates': 'Road geometry centerline; no elevation. Between-frame animation is interpolated.',
                                 'travel_time': 'Observed duration only; insertion/arrival may be outside recording.'})
    manifest['outputs'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(output.resolve() / 'index.html')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, required=True)
    parser.add_argument('--routes', type=Path, required=True)
    parser.add_argument('--trajectories', type=Path, required=True)
    parser.add_argument('--vehicles', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export(args.network, args.routes, args.trajectories, set(args.vehicles), args.output)


if __name__ == '__main__':
    main()
