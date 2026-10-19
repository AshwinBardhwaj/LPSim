#!/usr/bin/env python3
"""Separate, cold-cache Nsight diagnostics; never used to select schedules."""
from pathlib import Path
import subprocess,json,os
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/vehicle-allocation-locality-20261018';B=R/'artifacts/current/berkeley-allocation-workloads-20261018';N=R/'data/networks/berkeley_junction_study'
METRICS=['gpu__time_duration.sum','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__thread_inst_executed_per_inst_executed.ratio','smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum','lts__t_sectors_op_read.sum','launch__registers_per_thread']
def main():
 assert json.loads((A/'complete.json').read_text())['passed']
 d=A/'profiles';d.mkdir(exist_ok=True)
 s=(R/'tools/vehicle_locality_replay.cu').read_text().replace('#define LPSIM_SCALABLE_LIBRARY','#include <cuda_profiler_api.h>\n#define LPSIM_SCALABLE_LIBRARY').replace('../src/simulator/scalable_junction_simulator.cu',str(R/'src/simulator/scalable_junction_simulator.cu')).replace('../artifacts/current/vehicle-allocation-locality-20261018/advance_tasks.cuh',str(A/'advance_tasks.cuh'))
 mark=' cudaEvent_t events[5];'
 assert mark in s
 s=s.replace(mark,' if(const char* selected=std::getenv("LPSIM_PROFILE_POLICY")){int p=atoi(selected);if(p<0||p>6)throw std::runtime_error("Invalid profile policy");pre();check(cudaDeviceSynchronize());check(cudaProfilerStart());advanceRun(p==0,fixed[p]);check(cudaDeviceSynchronize());check(cudaProfilerStop());checkCars(expectedMotion);return 0;}\n'+mark)
 (d/'profile_replay.cu').write_text(s)
 cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_87','-std=c++17',str(d/'profile_replay.cu'),'-o',str(d/'replay')];(d/'build.json').write_text(json.dumps(cmd));subprocess.run(cmd,check=True)
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True)
 settings=d/'clocks.conf';assert not settings.exists();meta={'before':clock('--show')};clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show')
  for case in ['distributed_16384_seed1802','concentrated_16384_seed1802']:
   for p in [0,1,3,5,6]:
    out=d/case/str(p);out.mkdir(parents=True,exist_ok=True)
    cmd=['sudo','-n','env','LPSIM_MOTION_CONTEXT=3',f'LPSIM_PROFILE_POLICY={p}','/usr/local/cuda/bin/ncu','--profile-from-start','off','--cache-control','all','--clock-control','none','--metrics',','.join(METRICS),'--export',str(out/'metrics'),'--force-overwrite',str(d/'replay'),str(N/'model.txt'),str(B/'runs'/case/'snapshot_300.bin'),str(N/'regions.txt'),str(out/'unused.csv'),'0'];(out/'command.json').write_text(json.dumps(cmd))
    with (out/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (out/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(out/'metrics.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
    print('profile',case,p,'passed',flush=True)
  (d/'complete.json').write_text('{"passed":true}')
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');(d/'environment.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
