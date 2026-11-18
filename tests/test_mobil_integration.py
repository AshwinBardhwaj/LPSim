#!/usr/bin/env python3
"""Run actual lane changes with 1/2/3 road-owner partitions and exact oracle checks.

Requires the resident binary built by gpu_resident_exchange/build.sh. Default is
three physical CUDA devices; --logical validates partitions on one device only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def write_fixture(root):
    roads = [(0, 1, 2, 500, 30), (1, 2, 2, 500, 30), (2, 3, 2, 500, 30)]
    moves = [(edge + 1, edge, edge + 1, lane, 0, 0, 0, 20, 0)
             for edge in (0, 1) for lane in (0, 1)]
    conflicts = [(a, b) for a in range(4) for b in range(4) if moves[a][0] == moves[b][0]]
    model = ['3 4 4', *(' '.join(map(str, r)) for r in roads), *(' '.join(map(str, m)) for m in moves),
             f'SPARSE {len(conflicts)}', *(f'{a} {b}' for a, b in conflicts)]
    (root / 'model.txt').write_text('\n'.join(model) + '\n')
    records = []
    for road in (0, 1):
        for position, speed in ((100, 15), (120, 5)):
            edges = [road, road + 1] + [0] * 30
            lanes = [0] * 32
            movements = [road * 2] + [0] * 31
            records.append(struct.pack('<102i5f', 1, 0, 2, -1, 0, 0, *edges, *lanes, *movements,
                                       0, position, speed, 0, 0))
    (root / 'snapshot.bin').write_bytes(struct.pack('<if', len(records), 0) + b''.join(records))
    for count, owners in ((1, [0, 0, 0, 0]), (2, [0, 1, 0, 1]), (3, [0, 1, 2, 0])):
        (root / f'owners-{count}.txt').write_text('\n'.join(map(str, owners)) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--logical', action='store_true')
    p.add_argument('--steps', type=int, default=80)
    args = p.parse_args()
    if args.steps < 1:
        p.error('steps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    write_fixture(args.output)
    env = {k: v for k, v in os.environ.items() if not k.startswith('LPSIM_MOBIL')}
    env.pop('LPSIM_LOGICAL', None)
    if args.logical:
        env['LPSIM_LOGICAL'] = '1'
    expected = {}; rows = []
    for enabled in (0, 1):
        env['LPSIM_MOBIL'] = str(enabled)
        for ranks in (1, 2, 3):
            for schedule in ('barrier', 'neighbor'):
                for rebuild in (0, 1):
                    env['LPSIM_SLOT_REBUILD'] = str(rebuild)
                    label = f'mobil{enabled}-r{ranks}-{schedule}-rebuild{rebuild}'
                    out = args.output / label
                    cmd = [str(args.binary.resolve()), str(args.output / 'model.txt'),
                           str(args.output / 'snapshot.bin'), str(args.output / f'owners-{ranks}.txt'),
                           str(args.steps), str(out), str(ranks), '1', schedule, '0']
                    with (args.output / (label + '.log')).open('w') as log:
                        subprocess.run(cmd, env=env, check=True, timeout=180, stdout=log, stderr=subprocess.STDOUT)
                    metrics = json.loads((out / 'metrics.json').read_text())
                    digest = hashlib.sha256((out / 'final.bin').read_bytes()).hexdigest()
                    if enabled and metrics['mobil_changes'] <= 0:
                        raise AssertionError('MOBIL fixture did not change any lane')
                    if not enabled and metrics['mobil_changes'] != 0:
                        raise AssertionError('Disabled MOBIL changed lanes')
                    result = (digest, metrics['mobil_changes'])
                    if enabled in expected and expected[enabled] != result:
                        raise AssertionError(f'Partition/scheduling/slot-layout mismatch: {label}')
                    expected[enabled] = result
                    rows.append({'case': label, 'sha256': digest, 'changes': metrics['mobil_changes']})
                    (args.output / 'partial.json').write_text(json.dumps(rows, indent=2) + '\n')
    if expected[0][0] == expected[1][0]:
        raise AssertionError('Enabled and disabled fixture outcomes unexpectedly identical')
    receipt = {'passed': True, 'logical': args.logical, 'steps': args.steps,
               'verified_runs': len(rows), 'binary_sha256': hashlib.sha256(args.binary.read_bytes()).hexdigest(), 'results': rows}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(f'PASS {len(rows)} runs, {len(rows)*args.steps} exact timesteps; logical={args.logical}')


if __name__ == '__main__':
    main()
