#!/usr/bin/env python3
"""Compare serial and parallel junction scheduling without changing traffic behavior."""
import os,json,subprocess,hashlib,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'artifacts/junction_platoons_20261007';OUT=ROOT/'artifacts/junction_parallel_20261007'
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',action='append');p.add_argument('--repeat',type=int,default=1);p.add_argument('--exclusive',type=int,nargs='+',choices=[0,1],default=[0,1]);args=p.parse_args();records=[]
 for exclusive in args.exclusive:
  for case in sorted(SRC.glob('*/case.json')):
   name=case.parent.name
   if args.case and name not in args.case:continue
   for repeat in range(args.repeat):
    results={}
    for parallel in ((0,1) if repeat%2==0 else (1,0)):
     folder=OUT/f'exclusive_{exclusive}'/name/f'repeat_{repeat}'/('parallel' if parallel else 'serial');folder.mkdir(parents=True,exist_ok=True)
     env=dict(os.environ,LPSIM_JUNCTION_SCENARIO=str(case.parent/'scenario.txt'),LPSIM_JUNCTION_PARALLEL=str(parallel),LPSIM_JUNCTION_EXCLUSIVE=str(exclusive))
     start=time.perf_counter()
     with (folder/'run.log').open('w') as f:r=subprocess.run([str(ROOT/'build/lpsim')],cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT)
     assert r.returncode==0,str(folder/'run.log')
     m=json.loads((folder/'metrics.json').read_text());m.update(wall_seconds=time.perf_counter()-start,case=name,repeat=repeat)
     results[parallel]=m;records.append(m)
    for key in ('completed','active','pending','gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples','entries','right_on_red_entries','downstream_blocked_decisions','yield_blocked_decisions','max_same_movement_occupancy'):assert results[0][key]==results[1][key],(name,key,results)
    hashes={}
    parent=folder.parent
    for file in ('states.csv','events.csv','timeseries.csv','signals.csv'):
     digest=[hashlib.sha256((parent/mode/file).read_bytes()).hexdigest() for mode in ('serial','parallel')];assert digest[0]==digest[1],(name,file);hashes[file]=digest[0]
    (parent/'equivalence.json').write_text(json.dumps(dict(passed=True,identical_files=hashes)))
    print(exclusive,name,repeat,'identical; junction ms',results[0]['junction_gpu_ms'],results[1]['junction_gpu_ms'],flush=True)
    (OUT/('comparison_'+('_'.join(args.case) if args.case else 'all')+'.json')).write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
