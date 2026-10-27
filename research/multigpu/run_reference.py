"""Portable frozen reference invocation from the same binary snapshot inputs."""
from pathlib import Path
import argparse, json, os, subprocess
from prepare import read_snapshot
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('case');p.add_argument('output',type=Path);p.add_argument('--index',choices=['bulk','incremental'],default='bulk');p.add_argument('--steps',type=int,default=2400);a=p.parse_args()
b=a.bundle.resolve();d=a.output.resolve();d.mkdir(parents=True,exist_ok=False);snapshot=b/'inputs'/a.case/'snapshot.bin';t,cars=read_snapshot(snapshot)
model=b/'inputs/network/model.txt';lines=[str(model),f'{len(cars)} {a.steps*.25} .25 0']
for c in cars:
    n=c[2];lines.append(' '.join(map(str,[c[102],n,*c[6:6+n],*c[38:38+n],*c[70:70+n-1],-1])))
scenario=d/'scenario.txt';scenario.write_text('\n'.join(lines)+'\n')
env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_COMPACT_MOTION='1',LPSIM_LEADER_INDEX='1',LPSIM_LEADER_SORT='radix',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1',LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_TRAJECTORY_STRIDE='120',LPSIM_SAVE_SNAPSHOTS='1',LPSIM_RESTART=str(snapshot),LPSIM_INDEX_POLICY=a.index)
cmd=[str(b/'bin'/('reference' if a.index=='bulk' else 'reference_incremental')),str(scenario)];(d/'command.json').write_text(json.dumps(dict(argv=cmd,env={k:v for k,v in env.items() if k.startswith('LPSIM_')}),indent=2)+'\n')
with (d/'run.log').open('w') as log:subprocess.run(cmd,cwd=d,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
