#!/usr/bin/env python3
"""Freeze the physics closure and generate narrowly scoped stable-ID adapters."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / 'artifacts/multigpu-preparation-20261027/source'


def function(text, name):
    start = text.index('__global__ void ' + name + '(')
    brace = text.index('{', start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    return text[start:end]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Frozen source contract changed: {old}')
    return text.replace(old, new)


def adapters():
    sim = (FROZEN / 'src/simulator/scalable_junction_simulator.cu').read_text()
    compact = (FROZEN / 'tools/compact_motion.cuh').read_text()
    advance = function(compact, 'advanceCompact')
    advance = replace_once(advance, 'advanceCompact(', 'advanceSlots(const int* ids,')
    advance = replace_once(advance, 'j<chosenConnector', 'ids[j]<ids[chosenConnector]')
    summary = function(sim, 'summarizeJunctions')
    summary = replace_once(summary, 'summarizeJunctions(', 'summarizeSlots(const int* ids,')
    summary = replace_once(summary, 'c.lanes[0],i)', 'c.lanes[0],ids[i])')
    arbitration = function(sim, 'arbitrateIndexedJunctions')
    arbitration = replace_once(arbitration, 'arbitrateIndexedJunctions(',
                               'arbitrateSlots(const int* lookup,')
    arbitration = replace_once(arbitration, 'int i=static_cast<unsigned int>(keys[k]);',
                               'int i=lookup[static_cast<unsigned int>(keys[k])];')
    return '// Generated from frozen physics; only identity handling changes.\n' + '\n\n'.join(
        [advance, summary, arbitration]) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(FROZEN, out / 'source')
    study = Path(__file__).resolve().parent
    shutil.copytree(study, out / 'study', ignore=shutil.ignore_patterns('__pycache__'))
    baseline = ROOT / 'research/multigpu/exchange_study'
    (out / 'baseline').mkdir()
    for name in ('local.cu', 'trace.cuh'):
        shutil.copy2(baseline / name, out / 'baseline' / name)
    shutil.copy2(ROOT / 'research/multigpu/a100_study/concurrent.cu', out / 'baseline/original.cu')
    shutil.copy2(ROOT / 'src/simulator/mobil.cuh', out / 'study/mobil.cuh')
    (out / 'study/physics_adapters.cuh').write_text(adapters())
    files = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(out.rglob('*')) if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(files, indent=2) + '\n')
    print(out)


if __name__ == '__main__':
    main()
