#!/usr/bin/env python3
"""Installed as rebuild.py at the root of a portable lane-review package."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


for name, entry in manifest['inputs'].items():
    if digest(root / name) != entry['sha256']:
        raise SystemExit(f'Captured input changed: {name}')
subprocess.run([sys.executable, str(root / 'code/process_routes.py'),
    '--network', str(root / 'inputs/network'), '--routes', str(root / 'inputs/routes.csv'),
    '--trajectories', str(root / 'inputs/trajectories.csv'), '--output', str(root / 'trips.json'),
    '--lane-width', str(manifest['lane_width_m'])], check=True)
checksums = manifest.setdefault('artifact_sha256', {})
for name in list(checksums):
    if name.startswith('playback/'):
        del checksums[name]
playback = json.loads((root / 'playback/manifest.json').read_text())
for name in ['trips.json', 'trips.diagnostics.json', 'playback/manifest.json',
             'playback/' + playback['lane_changes'],
             *['playback/' + path for path in playback['chunks'].values()]]:
    checksums[name] = digest(root / name)
manifest['diagnostics'] = json.loads((root / 'trips.diagnostics.json').read_text())
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2))
