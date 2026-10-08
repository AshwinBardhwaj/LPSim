#!/usr/bin/env python3
"""Run the opt-in physical junction CUDA backend and audit its movement events."""
import argparse,csv,json,os,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_platoons_20261007'
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',action='append');p.add_argument('--exclusive',action='store_true');args=p.parse_args()
 cases=sorted(ART.glob('*/case.json'),key=lambda p:(p.parent.name.startswith('network'),p.parent.name))
 for file in cases:
  folder=file.parent
  if args.case and folder.name not in args.case:continue
  if args.exclusive:
   original=folder;folder=ART/'exclusive'/original.name;folder.mkdir(parents=True,exist_ok=True)
   import shutil
   for name in ('scenario.txt','case.json','demand.json'):shutil.copy2(original/name,folder/name)
  env=dict(os.environ,LPSIM_JUNCTION_PARALLEL="0",LPSIM_JUNCTION_SCENARIO=str(folder/'scenario.txt'))
  env['LPSIM_JUNCTION_EXCLUSIVE']='1' if args.exclusive else '0'
  with (folder/'run.log').open('w') as log:r=subprocess.run([str(ROOT/'build/lpsim')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
  (folder/'execution.json').write_text(json.dumps(dict(returncode=r.returncode,binary_sha256=hashlib.sha256((ROOT/'build/lpsim').read_bytes()).hexdigest())))
  print(folder.name,r.returncode,flush=True)
  if r.returncode:raise RuntimeError(str(folder/'run.log'))
  m=json.loads((folder/'metrics.json').read_text());events=list(csv.DictReader((folder/'events.csv').open()))
  assert all(not m[k] for k in ('gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'))
  assert m['trips']==m['active']+m['completed']+m['pending']
  assert all(int(e['light'])==2 or (int(e['event'])==2 and int(e['turn'])==-1 and int(e['light'])==0 and float(e['wait_seconds'])>=2) for e in events)
  if folder.name in ('through_red','left_red'):assert m['entries']==0
  if folder.name=='right_on_red':assert m['right_on_red_entries']==1
  if folder.name in ('through_green','left_green'):assert m['completed']==1
  if folder.name=='blocked_exit':assert m['downstream_blocked_decisions']>0 and not events
  if folder.name=='right_yields':assert m['yield_blocked_decisions']>0 and m['right_on_red_entries']==1 and m['completed']==2
  if folder.name=='same_movement_platoon':
   assert m['completed']==12
   if not args.exclusive:assert m['max_same_movement_occupancy']>=2
  (folder/'validation.json').write_text(json.dumps(dict(passed=True,entry_events_checked=len(events))))
if __name__=='__main__':main()
