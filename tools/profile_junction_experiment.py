#!/usr/bin/env python3
"""Collect isolated Nsight Compute samples; profiling outputs never replace normal runs."""
import os,json,subprocess,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_platoons_20261007'
METRICS=['gpu__time_duration.sum','l1tex__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed','smsp__sass_average_branch_targets_threads_uniform.pct','smsp__warp_issue_stalled_wait_per_warp_active.pct','smsp__warp_issue_stalled_not_selected_per_warp_active.pct','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__thread_inst_executed_per_inst_executed.ratio','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']
def main():
 p=argparse.ArgumentParser();p.add_argument('--trips',type=int,nargs='+',default=[20,400,1600]);p.add_argument('--times',type=int,nargs='+',default=[60,180,450]);p.add_argument('--modes',nargs='+',default=['exclusive','platoon']);args=p.parse_args()
 for mode in args.modes:
  for n,t in [(n,t) for n in args.trips for t in args.times]:
   folder=ART/'profiles'/f'{mode}_{n}_{t}';folder.mkdir(parents=True,exist_ok=True)
   lines=(ART/f'network_{n}'/'scenario.txt').read_text().splitlines();line=lines[1].split();line[1]=str(t+1);lines[1]=' '.join(line);(folder/'scenario.txt').write_text('\n'.join(lines)+'\n')
   cmd=['sudo','-n','env','LPSIM_JUNCTION_PARALLEL=0',f'LPSIM_JUNCTION_SCENARIO={folder}/scenario.txt',f'LPSIM_JUNCTION_EXCLUSIVE={int(mode=="exclusive")}','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS),'--launch-skip',str(t*8),'--launch-count','4','--target-processes','all','--force-overwrite','--export',str(folder/'kernels'),str(ROOT/'build/lpsim')]
   (folder/'command.json').write_text(json.dumps(cmd,indent=2))
   with (folder/'ncu.log').open('w') as f:r=subprocess.run(cmd,cwd=folder,stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError(f'{folder}: ncu exited {r.returncode}')
   with (folder/'kernels.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'kernels.ncu-rep'),'--csv','--page','raw'],stdout=f,check=True)
   (folder/'provenance.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256((ROOT/'build/lpsim').read_bytes()).hexdigest(),sample_time_seconds=t,launch_count=4,mode=mode,requested_trips=n)))
   print(mode,n,t,'profile complete',flush=True)
if __name__=='__main__':main()
