"""Reproduce paired immutable snapshot tests and full-run equivalence."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['snapshots','trajectories'],required=True);p.add_argument('--output',type=Path,default=ROOT/'artifacts/current/memory-policy-20261022');p.add_argument('--baseline',type=Path,default=ROOT/'artifacts/current/berkeley-block-imbalance-20261019');args=p.parse_args()
a=args.output.resolve();b=args.baseline.resolve()
cases=json.loads((b/'prepared.json').read_text())['cases']
env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_COMPACT_MOTION='1',LPSIM_LEADER_INDEX='1',LPSIM_LEADER_SORT='radix',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1',LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_SAVE_SNAPSHOTS='1',LPSIM_TRAJECTORY_STRIDE='120')
if args.phase=='snapshots':
 (a/'replays').mkdir(exist_ok=True)
 for batch in range(3):
  for c in cases:
   for t in [60,180,300,450]:
    output=a/'replays'/f"{c['name']}_{t}_{batch}.csv"
    cmd=[str(a/'replay'),str(ROOT/'data/networks/berkeley_junction_study/model.txt'),str(b/'runs'/c['name']/f'snapshot_{t}.bin'),str(ROOT/'data/networks/berkeley_junction_study/regions.txt'),str(output),'0']
    subprocess.run(cmd,env=env,check=True)
   print(batch,c['name'],flush=True)
else:
 env.pop('LPSIM_LEADER_SORT',None) # Exercise the production default.
 results=[]
 for c in cases:
  d=a/'runs'/c['name'];d.mkdir(parents=True,exist_ok=True)
  cmd=[str(a/'compact_simulator'),str(b/'cases'/c['name']/'scenario.txt')]
  (d/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith('LPSIM_')}),indent=2))
  start=time.monotonic()
  with (d/'run.log').open('w') as log:subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  base=b/'runs'/c['name'];names=['states.csv','events.csv','signals.csv','timeseries.csv']+[x.name for x in base.glob('snapshot_*.bin')]
  checks={name:hashlib.sha256((d/name).read_bytes()).hexdigest()==hashlib.sha256((base/name).read_bytes()).hexdigest() for name in names}
  results.append(dict(case=c['name'],elapsed_s=time.monotonic()-start,checks=checks,metrics=json.loads((d/'metrics.json').read_text())))
  (a/'trajectory_equivalence.json').write_text(json.dumps(results,indent=2))
  print(c['name'],all(checks.values()),flush=True)
  if not all(checks.values()):raise RuntimeError('Trajectory mismatch')
