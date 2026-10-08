#!/usr/bin/env python3
"""Matched unprofiled timing replicates; full 600 s scenarios, audits kept enabled."""
from pathlib import Path
import os,json,subprocess,time,hashlib
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_platoons_20261007'
def main():
 records=[]
 for repeat in range(3):
  for n in (20,400,1600):
   for mode in (('exclusive','platoon') if repeat%2==0 else ('platoon','exclusive')):
    src=ART/f'network_{n}';folder=ART/'timing'/f'{mode}_{n}_{repeat}';folder.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,LPSIM_JUNCTION_PARALLEL="0",LPSIM_JUNCTION_SCENARIO=str(src/'scenario.txt'),LPSIM_JUNCTION_EXCLUSIVE=str(int(mode=='exclusive')))
    start=time.perf_counter()
    with (folder/'run.log').open('w') as log:p=subprocess.run([str(ROOT/'build/lpsim')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    wall=time.perf_counter()-start
    if p.returncode:raise RuntimeError(str(folder))
    m=json.loads((folder/'metrics.json').read_text());m.update(mode=mode,repeat=repeat,wall_seconds=wall,binary_sha256=hashlib.sha256((ROOT/'build/lpsim').read_bytes()).hexdigest());records.append(m)
    # Keep metrics/logs; full trajectories already exist in the audited main runs.
    for name in ('states.csv','signals.csv','events.csv','timeseries.csv'):(folder/name).unlink()
    (ART/'timing.json').write_text(json.dumps(records,indent=2));print(mode,n,repeat,round(wall,2),flush=True)
if __name__=='__main__':main()
