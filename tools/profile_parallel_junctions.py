#!/usr/bin/env python3
"""Matched kernel samples for serial versus parallel following, at t=180 s."""
import json,subprocess
from pathlib import Path
from profile_junction_experiment import METRICS
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_parallel_20261007'
def main():
 for parallel in (0,1):
  folder=ART/'profiles'/('parallel' if parallel else 'serial');folder.mkdir(parents=True,exist_ok=True)
  lines=(ROOT/'artifacts/junction_platoons_20261007/network_1600/scenario.txt').read_text().splitlines();fields=lines[1].split();fields[1]='181';lines[1]=' '.join(fields);scenario=folder/'scenario.txt';scenario.write_text('\n'.join(lines)+'\n')
  kernels=6 if parallel else 2
  cmd=['sudo','-n','env',f'LPSIM_JUNCTION_SCENARIO={scenario}',f'LPSIM_JUNCTION_PARALLEL={parallel}','LPSIM_JUNCTION_EXCLUSIVE=0','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS),'--launch-skip',str(180*4*kernels),'--launch-count',str(2*kernels),'--force-overwrite','--export',str(folder/'kernels'),str(ROOT/'build/lpsim')]
  (folder/'command.json').write_text(json.dumps(cmd,indent=2))
  with (folder/'ncu.log').open('w') as f:subprocess.run(cmd,cwd=folder,stdout=f,stderr=subprocess.STDOUT,check=True)
  with (folder/'kernels.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'kernels.ncu-rep'),'--csv','--page','raw'],stdout=f,check=True)
  print('Profile complete:',folder.name,flush=True)
if __name__=='__main__':main()
