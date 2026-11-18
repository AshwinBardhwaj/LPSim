"""Generate candidate maps and proxy diagnostics; never launch Modal."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import struct
import time

from policies import POLICIES, build_features, describe, partition

ROOT = Path(__file__).resolve().parents[3]
CAR = struct.Struct('<102i5f')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', nargs='+', default=['berkeley_8192', 'sf_98304'])
    p.add_argument('--parts', nargs='+', type=int, default=[2, 3])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = ROOT/'artifacts/local-scaling-20261028/inputs'
    known = {c['name']: c for c in json.loads((inputs/'cases.json').read_text())}
    rows, manifest = [], []
    for name in args.cases:
        case = known[name]
        geometry_file = ROOT/'data/networks'/case['network']/'geometry.json'
        snapshot = inputs/name/'snapshot.bin'
        raw = snapshot.read_bytes()
        n, t = struct.unpack_from('<if', raw)
        if n < 1 or len(raw) != 8+n*CAR.size:
            raise ValueError('Snapshot ABI mismatch')
        cars = [CAR.unpack_from(raw, 8+i*CAR.size) for i in range(n)]
        geo = json.loads(geometry_file.read_text())
        started = time.perf_counter()
        features = build_features(geo, cars, expected_active=n)
        feature_ms = 1000*(time.perf_counter()-started)
        folder = args.output/name
        folder.mkdir()
        (folder/'prior.json').write_text(json.dumps(dict(
            expected_road_population=n, nominal_dt=.25,
            forecast='stationary free-flow route-residence proxy',
            ignored=['observed position', 'observed speed', 'observed state',
                     'departures', 'signals', 'future trajectories', 'connector dwell time'],
            feature_construction_ms=feature_ms), indent=2)+'\n')
        for parts in args.parts:
            for policy in ('geography', 'paper_balanced', 'paper_community', *POLICIES):
                if policy in POLICIES:
                    owners, meta = partition(geo, cars, policy, parts, features=features)
                    target = folder/f'{policy}_{parts}.txt'
                    target.write_text('\n'.join(map(str, owners))+'\n')
                    meta['feature_construction_ms'] = feature_ms
                    meta['total_construction_ms'] = feature_ms+meta['construction_ms']
                    meta['map_sha256'] = sha(target)
                    # Exact repeat verifies deterministic assignment, excluding timers.
                    repeat, _ = partition(geo, cars, policy, parts, features=features)
                    if owners != repeat:
                        raise RuntimeError('Nondeterministic ownership map')
                    meta['repeat_map_identical'] = True
                    (folder/f'{policy}_{parts}.json').write_text(json.dumps(meta, indent=2)+'\n')
                else:
                    target = inputs/name/f'{policy}_{parts}.txt'
                    owners = list(map(int, target.read_text().split()))
                    meta = dict(total_construction_ms=None)
                stats = describe(owners, features, parts)
                rows.append(dict(case=name, parts=parts, policy=policy,
                                 construction_ms=meta['total_construction_ms'],
                                 expected_owned_max_mean=stats['owned_max_mean'],
                                 expected_ghosts=sum(stats['expected_ghosts']),
                                 cut_roads=stats['cut_roads'],
                                 boundary_byte_proxy_per_step=stats['boundary_objective'],
                                 directed_neighbor_pairs=stats['directed_neighbor_pairs'],
                                 measured_runtime_ms=None))
        manifest.append(dict(case=name, geometry=str(geometry_file.relative_to(ROOT)),
                             geometry_sha256=sha(geometry_file), snapshot=str(snapshot.relative_to(ROOT)),
                             snapshot_sha256=sha(snapshot), snapshot_time=t))
        print(f'Prepared {name}: route-only proxy; no runtime result', flush=True)
    with (args.output/'proxy-comparison.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (args.output/'manifest.json').write_text(json.dumps(dict(
        status='offline candidates; not performance validated',
        sources={p.name: sha(p) for p in Path(__file__).parent.glob('*.py')},
        cases=manifest, parts=args.parts, seed=2710, new_modal_spend_usd=0), indent=2)+'\n')


if __name__ == '__main__':
    main()
