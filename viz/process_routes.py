#!/usr/bin/env python3
"""Convert authoritative GPU samples to TripsLayer paths; never infer edge identity."""
import argparse
import bisect
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def distance(a, b):
    """Polyline arc metric only; simulation progress uses edges.csv.length."""
    dx = math.radians(b[0] - a[0]) * math.cos(math.radians((a[1] + b[1]) / 2))
    dy = math.radians(b[1] - a[1])
    return 6371000 * math.hypot(dx, dy)


@dataclass
class Edge:
    u: int
    v: int
    length: float
    points: list
    offsets: list
    lanes: int = 1
    two_way: bool = False

    def point(self, pos):
        i = min(bisect.bisect_right(self.offsets, pos) - 1, len(self.points) - 2)
        i = max(i, 0)
        span = self.offsets[i + 1] - self.offsets[i]
        f = (pos - self.offsets[i]) / span if span else 0
        a, b = self.points[i:i + 2]
        return [a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])]

    def lane_point(self, pos, lane, width=3.2):
        """Schematic lane centers: lane 0 is leftmost in travel direction."""
        point = self.point(pos)
        delta = min(0.1, self.length / 10)
        a = self.point(max(0, pos - delta))
        b = self.point(min(self.length, pos + delta))
        scale = math.cos(math.radians(point[1]))
        dx, dy = (b[0] - a[0]) * scale, b[1] - a[1]
        norm = math.hypot(dx, dy)
        offset = (lane + 0.5 if self.two_way else lane - (self.lanes - 1) / 2) * width
        if norm:
            point[0] += offset * dy / norm / (111320 * scale)
            point[1] -= offset * dx / norm / 111320
        return point


def load_edges(network):
    with (network / 'nodes.csv').open() as f:
        nodes = {int(r.get('index', i)): (float(r.get('lon', r.get('x'))),
                                        float(r.get('lat', r.get('y'))))
                 for i, r in enumerate(csv.DictReader(f))}
    edges = {}
    with (network / 'edges.csv').open() as f:
        for r in csv.DictReader(f):
            uid, u, v = (int(r[k]) for k in ('uniqueid', 'u', 'v'))
            length = float(r['length'])
            if not math.isfinite(length) or length <= 0:
                raise ValueError(f'Edge {uid}: invalid simulation length')
            a, b = nodes[u], nodes[v]
            points = [a, b]
            wkt = r.get('geometry', '').strip()
            if wkt:
                match = re.fullmatch(r'LINESTRING\s*\((.*)\)', wkt, re.I)
                if not match:
                    raise ValueError(f'Edge {uid}: expected WKT LINESTRING')
                points = [tuple(map(float, p.split())) for p in match[1].split(',')]
                if len(points) < 2 or any(len(p) != 2 or not all(map(math.isfinite, p)) for p in points):
                    raise ValueError(f'Edge {uid}: invalid geometry')
                if distance(points[-1], a) + distance(points[0], b) < distance(points[0], a) + distance(points[-1], b):
                    points.reverse()
                if distance(points[0], a) > 2 or distance(points[-1], b) > 2:
                    raise ValueError(f'Edge {uid}: geometry endpoints disagree with topology')
                points[0], points[-1] = a, b
            arc = [0.0]
            for x, y in zip(points, points[1:]):
                arc.append(arc[-1] + distance(x, y))
            if arc[-1] <= 0:
                raise ValueError(f'Edge {uid}: degenerate geometry')
            edges[uid] = Edge(u, v, length, points, [s / arc[-1] * length for s in arc])
    pairs = {(e.u, e.v) for e in edges.values()}
    with (network / 'edges.csv').open() as f:
        for r in csv.DictReader(f):
            e = edges[int(r['uniqueid'])]
            e.lanes = max(1, int(float(r.get('lanes', 1))))
            e.two_way = (e.v, e.u) in pairs
    return edges


def load_routes(filename):
    routes = {}
    with filename.open() as f:
        for line in f:
            if not line.strip() or line.startswith('p:'):
                continue
            match = re.fullmatch(r'\s*(\d+):\[(.*?)\]:[^\n]+\s*', line)
            if not match:
                raise ValueError(f'Malformed route: {line[:100]}')
            # The final field is route distance, not departure time.
            routes[match[1]] = [int(e) for e in match[2].split(',') if e.strip()]
    return routes


def convert(network, routes_file, trajectories, max_speed=45, lane_width=0, sink=None):
    edges, routes = load_edges(network), load_routes(routes_file)
    rows, stats = defaultdict(list), Counter()
    database = tempfile.NamedTemporaryFile(suffix='.sqlite') if sink else None
    connection = sqlite3.connect(database.name) if database else None
    if connection:
        connection.execute('CREATE TABLE samples (vid TEXT, t REAL, idx INTEGER, eid INTEGER, pos REAL, lane INTEGER)')
        connection.execute('PRAGMA cache_size=-16384')
        connection.execute('PRAGMA temp_store=FILE')

    with trajectories.open(encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        required = {'vehicle_id', 'timestamp', 'edge_id', 'route_index', 'pos_m', 'edge_id_kind'}
        if not required <= set(reader.fieldnames or []):
            raise ValueError('Legacy/ambiguous trajectory schema. Re-run the patched simulator with '
                             'LPSIM_TRAJECTORIES=trajectories_physical.csv; internal lane IDs cannot '
                             'safely be interpreted as physical edge IDs.')
        if lane_width and 'lane_idx' not in reader.fieldnames:
            raise ValueError('Lane visualization requires recorded lane_idx')
        for r in reader:
            if r['edge_id_kind'] != 'uniqueid':
                raise ValueError('Trajectory edge_id_kind must be uniqueid')
            t, pos = float(r['timestamp']), float(r['pos_m'])
            if not math.isfinite(t) or not math.isfinite(pos):
                raise ValueError('Non-finite trajectory timestamp or position')
            record = (t, int(r['route_index']), int(r['edge_id']), pos, int(r.get('lane_idx', 0)))
            if connection:
                connection.execute('INSERT INTO samples VALUES (?,?,?,?,?,?)', (r['vehicle_id'], *record))
            else:
                rows[r['vehicle_id']].append(record)
            stats['input_samples'] += 1
    trips = []
    if connection:
        connection.commit()
        connection.execute('CREATE INDEX sample_order ON samples(vid,t)')
        groups = ((vid, [row[1:] for row in group]) for vid, group in
                  itertools.groupby(connection.execute('SELECT * FROM samples ORDER BY vid,t'), key=lambda row: row[0]))
    else:
        groups = rows.items()
    for vid, samples in groups:
        route = routes.get(vid)
        if not route or any(e not in edges for e in route):
            stats['missing_routes_or_edges'] += 1
            continue
        prefix = [0.0]
        for eid in route:
            prefix.append(prefix[-1] + edges[eid].length)
        path, details, previous = [], [], None

        def flush():
            nonlocal path, details, previous
            if len(path) >= 2:
                trip = {'vendor': str(vid), 'path': path, 'samples': details}
                if sink:
                    sink(trip)
                else:
                    trips.append(trip)
                stats['output_segments'] += 1
            path, details, previous = [], [], None

        for t, group in itertools.groupby(sorted(samples), key=lambda s: s[0]):
            group = list(group)
            unique = set(group)
            stats['duplicate_samples'] += len(group) - len(unique)
            if len(unique) != 1:
                stats['conflicting_timestamps'] += 1
                flush()
                continue
            _, idx, eid, pos, lane = unique.pop()
            if idx < 0 or idx >= len(route):
                stats['route_overflow'] += 1
                flush()
                break
            if eid != route[idx]:
                stats['route_mismatch'] += 1
                flush()
                continue
            edge = edges[eid]
            if lane_width and not 0 <= lane < edge.lanes:
                stats['invalid_lane_samples'] += 1
                flush()
                continue
            if pos < -0.01 or pos > edge.length + 0.01:
                stats['position_out_of_bounds'] += 1
                flush()
                continue
            pos = min(edge.length, max(0, pos))
            cumulative = prefix[idx] + pos
            if previous:
                pt, pi, ps, previous_lane = previous
                if pi == idx and previous_lane != lane:
                    stats['recorded_lane_changes'] += 1
                ds = cumulative - ps
                connected = all(edges[route[j]].v == edges[route[j+1]].u for j in range(pi, idx))
                if idx < pi or ds < -0.01 or ds / (t - pt) > max_speed or not connected:
                    stats['kinematic_or_topology_splits'] += 1
                    flush()
                elif ds > 0:
                    # Preserve every measured endpoint. Extra vertices only prevent
                    # straight chords across bends between consecutive samples.
                    for j in range(pi, idx + 1):
                        e = edges[route[j]]
                        for point, offset in zip(e.points, e.offsets):
                            s = prefix[j] + offset
                            if ps < s < cumulative:
                                vt = pt + (t - pt) * (s - ps) / ds
                                if vt > path[-1][2]:
                                    interpolated_lane = previous_lane if j == pi else lane if j == idx else 0
                                    interpolated_lane = min(interpolated_lane, e.lanes - 1)
                                    if lane_width:
                                        point = e.lane_point(offset, interpolated_lane, lane_width)
                                    path.append([*point, vt])
                                    details.append(None) # Between-frame geometry, not an observation.

            point = edge.lane_point(pos, lane, lane_width) if lane_width else edge.point(pos)
            path.append([*point, t])
            details.append({'edge': eid, 'route_index': idx, 'lane': lane, 'pos_m': pos})
            previous = (t, idx, cumulative, lane)
            stats['accepted_samples'] += 1
        flush()
    if connection:
        connection.close()
        database.close()
    return trips, dict(stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--network', type=Path, default=ROOT / 'data/networks/berkeley')
    parser.add_argument('--routes', type=Path, default=ROOT / '0_route5to12.csv')
    parser.add_argument('--trajectories', type=Path, default=ROOT / 'trajectories_physical.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'viz/trips.json')
    parser.add_argument('--lane-width', type=float, default=0, help='Schematic lane spacing in metres; 0 preserves centerline rendering')
    parser.add_argument('--max-speed', type=float, default=45)
    args = parser.parse_args()
    if not math.isfinite(args.lane_width) or args.lane_width < 0:
        parser.error('--lane-width must be finite and nonnegative')
    if not math.isfinite(args.max_speed) or args.max_speed <= 0:
        parser.error('--max-speed must be positive and finite')
    try:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix('.json.tmp')
        with temporary.open('w') as out:
            out.write('[')
            first = True
            def emit(trip):
                nonlocal first
                if not first:
                    out.write(',')
                out.write(json.dumps(trip, separators=(',', ':'), allow_nan=False))
                first = False
            _, stats = convert(args.network, args.routes, args.trajectories, args.max_speed, args.lane_width, emit)
            out.write(']')
        if not stats.get('output_segments'):
            raise ValueError(f'No valid trajectory segments; existing output preserved. Diagnostics: {stats}')
        temporary.replace(args.output)
        args.output.with_suffix('.diagnostics.json').write_text(json.dumps(stats, indent=2) + '\n')
        from prepare_playback import prepare
        prepare(args.output)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(stats, indent=2))
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
