#!/usr/bin/env python3
"""Bounded, exact-gated local/physical evaluation; never provisions cloud GPUs."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--owners', type=Path, required=True)
    p.add_argument('--ranks', type=int, choices=(1, 2, 3), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--timeout', type=int, default=120)
    p.add_argument('--logical', action='store_true')
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    if os.environ.get("LPSIM_MOBIL", "0") != "0":
        p.error("Legacy baselines have no MOBIL; use tests/test_mobil_integration.py for the enabled-model comparison")
    if min(args.steps, args.repeats, args.timeout) < 1:
        p.error('steps, repeats and timeout must be positive')
    if args.logical and not args.verify_only:
        p.error('Logical partitions are correctness checks, not scaling measurements; use --verify-only')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.build / 'manifest.json').read_text())
    for name, digest in manifest.items():
        if sha(args.build / name) != digest:
            raise RuntimeError(f'Frozen source changed: {name}')
    receipt = {'inputs': {str(f.resolve()): sha(f) for f in (args.model, args.snapshot, args.owners)},
               'executables': {name: sha(args.build / name) for name in ('resident', 'baseline-local', 'baseline-original', 'protocol-tests')},
               'steps': args.steps, 'ranks': args.ranks, 'logical': args.logical,
               'complete': False, 'timing_runs': []}
    receipt_path = args.output / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    env = os.environ.copy()
    for key in ('LPSIM_LOGICAL', 'LPSIM_SLOT_REBUILD'):
        env.pop(key, None)
    if args.logical:
        env['LPSIM_LOGICAL'] = '1'
    env['LPSIM_EXCHANGE_MODE'] = '3'
    with (args.output / 'protocol.log').open('w') as log:
        subprocess.run([str(args.build / 'protocol-tests')], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True, timeout=args.timeout)

    variants = ('original-single', 'local-single', 'baseline', 'rebuild-barrier', 'slots-barrier', 'slots-neighbor')
    single_map = args.output / 'single-owners.txt'
    junction_count = int(args.model.read_text().splitlines()[0].split()[2])
    single_map.write_text('0\n' * junction_count)

    def run(variant, label, verify):
        out = args.output / label
        run_env = dict(env)
        common = [str(args.model), str(args.snapshot), str(args.owners), str(args.steps), str(out)]
        if variant == 'original-single':
            out.mkdir()
            cmd = [str(args.build / 'baseline-original'), *common, 'single', str(int(verify)), '0', '0']
        elif variant in ('baseline', 'local-single'):
            out.mkdir()
            if variant == 'local-single':
                common[2] = str(single_map)
            cmd = [str(args.build / 'baseline-local'), *common,
                   '1' if variant == 'local-single' else str(args.ranks), str(int(verify)), '0']
        else:
            run_env['LPSIM_SLOT_REBUILD'] = '1' if variant == 'rebuild-barrier' else '0'
            cmd = [str(args.build / 'resident'), *common, str(args.ranks), str(int(verify)),
                   'neighbor' if variant == 'slots-neighbor' else 'barrier', '0']
        with (args.output / (label + '.log')).open('w') as log:
            subprocess.run(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=args.timeout)
        metrics = json.loads((out / 'metrics.json').read_text())
        return sha(out / 'final.bin'), metrics

    expected = None
    for variant in variants:
        digest, _ = run(variant, 'verify-' + variant, True)
        if expected is not None and digest != expected:
            raise RuntimeError(f'Exact final hash mismatch: {variant}')
        expected = digest
    receipt['exact_gate_passed'] = True
    receipt['final_sha256'] = expected
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    rows = []
    if not args.verify_only:
        for variant in variants:
            digest, _ = run(variant, 'warmup-' + variant, False)
            if digest != expected:
                raise RuntimeError(f'Warmup mismatch: {variant}')
        rng = random.Random(20261105)
        for repeat in range(args.repeats):
            order = list(variants); rng.shuffle(order)
            for variant in order:
                digest, metrics = run(variant, f'timing-{repeat}-{variant}', False)
                if digest != expected:
                    raise RuntimeError(f'Timing mismatch: {variant}')
                ms = (metrics['compute_wall_ms'] if variant == 'original-single' else
                      metrics['compute_ms'] if variant in ('baseline', 'local-single') else metrics['wall_ms'])
                rows.append({'repeat': repeat, 'variant': variant, 'window_ms': ms, 'step_ms': ms / args.steps})
                receipt['timing_runs'] = rows
                receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        with (args.output / 'timings.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        medians = {v: statistics.median(r['step_ms'] for r in rows if r['variant'] == v) for v in variants}
        (args.output / 'medians.json').write_text(json.dumps(medians, indent=2) + '\n')
    receipt['complete'] = True
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(receipt_path)


if __name__ == '__main__':
    main()
