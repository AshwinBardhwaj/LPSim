#!/usr/bin/env python3
"""Verify captured inputs and artifacts against a lane-review SHA-256 manifest."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1] if len(sys.argv) > 1 else '.').resolve()
manifest = json.loads((root / 'manifest.json').read_text())
expected = {name: entry['sha256'] for name, entry in manifest['inputs'].items()}
expected.update(manifest['artifact_sha256'])
for name, checksum in expected.items():
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise SystemExit(f'Unsafe manifest path: {name}')
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    if h.hexdigest() != checksum:
        raise SystemExit(f'Checksum mismatch: {name}')
print(f'Verified {len(expected)} files')
