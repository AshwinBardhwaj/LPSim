#!/usr/bin/env python3
"""Build a portable lane-review site from an existing authoritative simulation run."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='New self-contained output directory')
    p.add_argument('--network', type=Path, default=ROOT / 'data/networks/berkeley')
    p.add_argument('--routes', type=Path, default=ROOT / '0_route5to12.csv')
    p.add_argument('--trajectories', type=Path, default=ROOT / 'trajectories_physical.csv')
    p.add_argument('--lane-width', type=float, default=3.2)
    args = p.parse_args()
    if not math.isfinite(args.lane_width) or args.lane_width <= 0:
        p.error('--lane-width must be positive and finite')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    network = inputs / 'network'
    network.mkdir(parents=True)
    for name in ('nodes.csv', 'edges.csv'):
        shutil.copy2(args.network / name, network / name)
    shutil.copy2(args.routes, inputs / 'routes.csv')
    shutil.copy2(args.trajectories, inputs / 'trajectories.csv')
    shutil.copy2(ROOT / 'data/command_line_options.ini', inputs / 'current_config.ini')
    for name in ('od_demand.csv', 'partitions.txt'):
        if (args.network / name).exists():
            shutil.copy2(args.network / name, network / name)
    code = out / 'code'
    code.mkdir()
    shutil.copy2(ROOT / 'tools/verify_lane_review.py', out / 'verify.py')
    shutil.copy2(Path(__file__), code / 'package_lane_review.py')
    shutil.copy2(ROOT / 'docs/GPU_SIGNAL_AUDIT.md', out / 'GPU_SIGNAL_AUDIT.md')
    for name in ('process_routes.py', 'prepare_playback.py'):
        shutil.copy2(ROOT / 'viz' / name, code / name)
    viewer = out / 'animated_sim'
    viewer.mkdir()
    shutil.copy2(ROOT / 'viz/animated_sim/index.html', viewer / 'index.html')
    (out / 'index.html').write_text('<!doctype html><meta http-equiv="refresh" content="0;url=animated_sim/">')
    spec = importlib.util.spec_from_file_location('lane_review_converter', code / 'process_routes.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    features = []
    for eid, edge in module.load_edges(network).items():
        for lane in range(edge.lanes):
            features.append({'type': 'Feature', 'properties': {'edge_id': eid, 'lane': lane},
                'geometry': {'type': 'LineString', 'coordinates': [edge.lane_point(s, lane, args.lane_width) for s in edge.offsets]}})
    (viewer / 'network.json').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}, separators=(',', ':')))
    shutil.copy2(ROOT / 'viz/rebuild_lane_review.py', out / 'rebuild.py')
    # Preserve current simulator source, including local tracked modifications and
    # the untracked snapshot header. This is current provenance, not proof of the
    # historical build used to create an existing CSV.
    source = out / 'simulator-source'
    source.mkdir()
    for name in ('src', 'include'):
        shutil.copytree(ROOT / name, source / name)
    shutil.copy2(ROOT / 'CMakeLists.txt', source / 'CMakeLists.txt')
    (out / 'working-tree.patch').write_bytes(subprocess.check_output(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT))
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
        'git_commit_at_packaging': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'lane_width_m': args.lane_width, 'lane_geometry': 'schematic; lane 0 leftmost; right-hand traffic',
        'historical_run_provenance': 'Existing CSV: current source/config captured at packaging; original build and seed not established',
        'renderer_dependencies': 'deck.gl and MapLibre loaded from the viewer CDN URLs; internet required',
        'inputs': {str(f.relative_to(out)): {'sha256': digest(f), 'bytes': f.stat().st_size} for f in inputs.rglob('*') if f.is_file()}}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Inputs captured. Building lane-aware playback with disk-backed sorting…', flush=True)
    subprocess.run([sys.executable, str(out / 'rebuild.py')], check=True)
    manifest['diagnostics'] = json.loads((out / 'trips.diagnostics.json').read_text())
    manifest['artifact_sha256'] = {str(f.relative_to(out)): digest(f) for folder in (code, source, viewer) for f in folder.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    manifest['artifact_sha256']['GPU_SIGNAL_AUDIT.md'] = digest(out / 'GPU_SIGNAL_AUDIT.md')
    manifest['artifact_sha256']['verify.py'] = digest(out / 'verify.py')
    manifest['artifact_sha256']['rebuild.py'] = digest(out / 'rebuild.py')
    manifest['artifact_sha256']['trips.json'] = digest(out / 'trips.json')
    playback = json.loads((out / 'playback/manifest.json').read_text())
    for relative in ['manifest.json', playback['lane_changes'], *playback['chunks'].values()]:
        manifest['artifact_sha256']['playback/' + relative] = digest(out / 'playback' / relative)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (out / 'README.md').write_text('''# Lane review package
Verify captured files with `python verify.py`.
Run `python -m http.server 8080 --directory .` here, then open `/animated_sim/`.
Rebuild playback with `python rebuild.py` (Python standard library only).
Inputs, converter code, viewer, current simulator source, working-tree patch,
and SHA-256 manifest are included. No original absolute file paths are required.
CDN libraries and map tiles require internet. The original simulator build,
random seed and configuration at simulation time cannot be recovered from CSV.

Lane width is schematic, not surveyed. Lane 0 is leftmost in travel direction.
Reverse links are drawn on opposite sides of their shared centerline. Separate
carriageways with no reverse link are centered independently. Curves are only as
accurate as the input geometry. Intersection lane connectors are not inferred.
Vehicle movement between samples is interpolated for playback: it is not a
measured lateral trajectory. Hover shows recorded lane values bracketing playback.
An index change within one edge is evidence of a recorded lane-state change;
an index change across edges may just be a lane-count reduction or reassignment.
Signals have not been patched or rerun by this packaging operation.
''')
    print(f'Ready: {out}')


if __name__ == '__main__':
    main()
