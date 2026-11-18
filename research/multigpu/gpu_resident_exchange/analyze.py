#!/usr/bin/env python3
"""Paired timing analysis; never treats logical-partition runs as scaling data."""
import argparse
import csv
import json
from pathlib import Path
import random
import statistics


def paired_speedup(baseline, candidate, samples=5000):
    if len(baseline) != len(candidate) or len(baseline) < 5:
        raise ValueError('At least five matched repetitions required')
    if any(x <= 0 for x in baseline + candidate):
        raise ValueError('Timings must be positive')
    rng = random.Random(20261105)
    values = []
    for _ in range(samples):
        indices = [rng.randrange(len(baseline)) for _ in baseline]
        values.append(statistics.median([baseline[i] for i in indices]) /
                      statistics.median([candidate[i] for i in indices]))
    values.sort()
    return {'median_speedup': statistics.median(baseline) / statistics.median(candidate),
            'paired_repeat_interval_95': [values[int(.025 * samples)], values[int(.975 * samples)]],
            'repetitions': len(baseline)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    receipt = json.loads((args.run / 'receipt.json').read_text())
    if receipt['logical'] or not receipt.get('complete') or not receipt.get('exact_gate_passed'):
        raise ValueError('A completed physical run with exact verification is required')
    with (args.run / 'timings.csv').open() as f:
        rows = list(csv.DictReader(f))
    values = {}
    for row in rows:
        v = values.setdefault(row['variant'], {})
        repeat = int(row['repeat'])
        if repeat in v:
            raise ValueError('Duplicate repetition')
        v[repeat] = float(row['step_ms'])
    results = {}
    for candidate in ('rebuild-barrier', 'slots-barrier', 'slots-neighbor'):
        results[candidate] = {}
        for baseline in ('original-single', 'local-single', 'baseline'):
            if values[candidate].keys() != values[baseline].keys():
                raise ValueError('Unmatched repetitions')
            order = sorted(values[baseline])
            results[candidate][baseline] = paired_speedup(
                [values[baseline][r] for r in order], [values[candidate][r] for r in order])
    report = {'comparisons': results,
              'scope': 'Paired timing repetitions on fixed demand; not held-out demand generalization.',
              'speedup_definition': 'baseline complete-step time / candidate complete-step time'}
    (args.run / 'comparisons.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
