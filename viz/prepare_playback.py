#!/usr/bin/env python3
"""Stream a TripsLayer JSON array into bounded time windows without downsampling."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
import tempfile


def iter_trips(filename):
    decoder = json.JSONDecoder()
    with Path(filename).open() as source:
        buffer, started, ended = '', False, False
        while True:
            block = source.read(1024 * 1024)
            buffer += block
            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != '[':
                        raise ValueError('Expected a JSON array')
                    buffer, started = buffer[1:], True
                    continue
                if buffer.startswith(','):
                    buffer = buffer[1:]
                    continue
                if buffer.startswith(']'):
                    ended = True
                    buffer = buffer[1:]
                    break
                if not buffer:
                    break
                try:
                    trip, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if not block:
                        raise
                    break
                yield trip
                buffer = buffer[end:]
            if ended:
                if buffer.strip() or source.read().strip():
                    raise ValueError('Trailing JSON data')
                return
            if not block:
                raise ValueError('Incomplete JSON array')


def prepare(filename, seconds=300):
    filename = Path(filename)
    target = filename.parent / 'playback'
    target.mkdir(exist_ok=True)
    generation = Path(tempfile.mkdtemp(prefix='run-', dir=target))
    events = (generation / 'lane_changes.csv').open('w', newline='')
    event_writer = csv.writer(events)
    event_writer.writerow(['vehicle_id', 'edge_id', 'route_index', 'time_before', 'time_after', 'lane_before', 'lane_after'])
    counts = {}
    minimum, maximum = math.inf, -math.inf
    for trip in iter_trips(filename):
        path = trip['path']
        if len(path) < 2:
            continue
        previous = None
        for point, sample in zip(path, trip.get('samples', [])):
            if sample is None:
                continue
            if previous:
                before_point, before = previous
                if sample['route_index'] == before['route_index'] and sample['lane'] != before['lane']:
                    event_writer.writerow([trip['vendor'], sample['edge'], sample['route_index'], before_point[2], point[2], before['lane'], sample['lane']])
            previous = point, sample
        times = [p[2] for p in path]
        minimum, maximum = min(minimum, times[0]), max(maximum, times[-1])
        for bucket in range(math.floor(times[0] / seconds), math.floor((times[-1] + 3) / seconds) + 1):
            start, end = bucket * seconds, (bucket + 1) * seconds
            lo = max(0, bisect.bisect_left(times, start - 3) - 1)
            hi = min(len(path), bisect.bisect_right(times, end) + 1)
            fragment = {'vendor': trip['vendor'], 'path': path[lo:hi]}
            if 'samples' in trip:
                fragment['samples'] = trip['samples'][lo:hi]
            with (generation / f'{bucket}.json').open('a') as out:
                out.write(',' if bucket in counts else '[')
                out.write(json.dumps(fragment, separators=(',', ':'), allow_nan=False))
            counts[bucket] = counts.get(bucket, 0) + 1
    events.close()
    if not counts:
        raise ValueError('No drawable trips')
    for bucket in counts:
        with (generation / f'{bucket}.json').open('a') as out:
            out.write(']')
    manifest = {'start': minimum, 'end': maximum, 'seconds': seconds, 'lane_changes': f'{generation.name}/lane_changes.csv',
                'chunks': {str(b): f'{generation.name}/{b}.json' for b in sorted(counts)}}
    temporary = target / 'manifest.json.tmp'
    temporary.write_text(json.dumps(manifest))
    temporary.replace(target / 'manifest.json')
    print(f'Prepared {len(counts)} playback windows in {generation}')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', nargs='?', type=Path, default=Path(__file__).with_name('trips.json'))
    args = parser.parse_args()
    prepare(args.input)
