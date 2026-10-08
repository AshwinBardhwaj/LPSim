#!/usr/bin/env python3
"""Check recorded occupancy invariants and actual signal phases after analysis."""
import argparse
import csv
import json
from pathlib import Path


def validate(folder):
    results = []
    for case in json.loads((folder / 'summary.json').read_text()):
        mode = case['signal_mode']
        phase_errors = 0
        for row in csv.DictReader((folder / case['case'] / 'signals.csv').open()):
            expected = mode == 'green' or (mode == 'cycle20' and int(float(row['step_start']) // 20) % 2 == 1)
            phase_errors += int(row['green']) != int(expected)
        checks = {
            'no_duplicate_cells': case['occupied_cell_duplicates'] == 0,
            'valid_samples': case['invalid_samples'] == 0,
            'no_red_crossings': case['red_stopline_violations'] == 0,
            'signal_phases': phase_errors == 0,
            'duration': case['steps'] == 1200,
            'trip_accounting': case['completed'] + case['active'] + case['pending'] == case['trips'],
        }
        if case['case'] == 'corridor_red':
            checks['held_at_red'] = case['completed'] == 0 and case['max_stopped_per_lane'] > 1
        if case['case'] in ('corridor_cycle', 'corridor_green'):
            checks['corridor_clears'] = case['completed'] == case['trips']
        results.append({'case': case['case'], 'checks': checks, 'passed': all(checks.values())})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    results = validate(args.folder)
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if results and all(r['passed'] for r in results) else 1)
