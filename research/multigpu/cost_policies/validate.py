"""Logical-device correctness gate; not an A100 performance benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifacts', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--steps', type=int, default=20)
    args = p.parse_args()
    results = []
    for policy in ('work_balanced', 'boundary_aware'):
        for parts in (2, 3):
            owner_file = args.artifacts/'berkeley_8192'/f'{policy}_{parts}.txt'
            output = args.artifacts/f'gate-{policy}-{parts}'
            output.mkdir(exist_ok=False)
            cmd = [str(args.binary.resolve()),
                   str(ROOT/'data/networks/berkeley_junction_study/model.txt'),
                   str(ROOT/'artifacts/local-scaling-20261028/inputs/berkeley_8192/snapshot.bin'),
                   str(owner_file.resolve()), str(args.steps), str(output.resolve()), str(parts), '1', '1']
            with (output/'run.log').open('w') as log:
                subprocess.run(cmd, env=dict(os.environ, LPSIM_LOGICAL='1'),
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
            m = json.loads((output/'metrics.json').read_text())
            if not m['verified']:
                raise RuntimeError('Physical verification disabled')
            results.append(dict(policy=policy, logical_partitions=parts,
                                steps=args.steps, passed=True,
                                owner_sha256=sha(owner_file), final_sha256=sha(output/'final.bin')))
            print(f'PASS {policy}/{parts}: {args.steps} exact-reference steps', flush=True)
    if len({r['final_sha256'] for r in results}) != 1:
        raise RuntimeError('Policies changed final physical state')
    (args.artifacts/'logical-validation.json').write_text(json.dumps(dict(
        device='one Jetson GPU with explicit logical messages',
        binary_sha256=sha(args.binary), tests=results, physical_A100_validation=False,
        performance_claim=False, total_exact_reference_steps=sum(r['steps'] for r in results)), indent=2)+'\n')


if __name__ == '__main__':
    main()
