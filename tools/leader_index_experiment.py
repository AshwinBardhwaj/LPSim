#!/usr/bin/env python3
"""Isolated leader-index ablation; historical fixtures and source remain untouched."""
import argparse, hashlib, json, shutil, subprocess, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/current/milestone2-snapshots-20261009'
ART=ROOT/'artifacts/current/milestone2-leader-index-20261014'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 assert not (ART/'manifest.json').exists(), 'Use a new artifact ID rather than overwriting results'
 src=(ROOT/'src/simulator/junction_simulator.cu').read_text()
 start=src.index('__global__ void advance');end=src.index('// Deterministic junction arbitration')
 kernel=src[start:end].replace('void advance(', 'void advanceIndexed(').replace('bool exclusive){','bool exclusive,const int* leaders){',1)
 old='for(int j=0;j<n;++j){if(j==i)continue;const Car& d=old[j];if(d.state==1&&d.edges[d.k]==edge&&d.lanes[d.k]==lane&&d.pos>c.pos){float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}'
 new='if(leaders[i]>=0){const Car& d=old[leaders[i]];float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}'
 assert kernel.count(old)==1
 kernel=kernel.replace(old,new)
 header=(ROOT/'tools/leader_index.cuh').read_text()
 # Thrust headers must be outside the production anonymous namespace.
 generated='#include "'+str(ROOT/'src/simulator/junction_simulator.cu')+'"\n'+header+'\n'+kernel
 (ART/'indexed_kernel.cuh').write_text(generated)
 harness=(ROOT/'tools/snapshot_benchmark.cu').read_text().replace('#include "../src/simulator/junction_simulator.cu"','#include "indexed_kernel.cuh"')
 harness=harness.replace('std::vector<float> times;','std::vector<float> times,indexTimes,motionTimes; int *orderDevice,*leaders;check(cudaMalloc(&orderDevice,n*sizeof(int)));check(cudaMalloc(&leaders,n*sizeof(int)));cudaEvent_t indexed;check(cudaEventCreate(&indexed));')
 launch='advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,180,.25,2,false);'
 replacement='initializeLaneOrder<<<(n+127)/128,128>>>(orderDevice,n);thrust::sort(thrust::device_pointer_cast(orderDevice),thrust::device_pointer_cast(orderDevice)+n,LanePositionLess{a});linkLaneLeaders<<<(n+127)/128,128>>>(a,orderDevice,leaders,n);check(cudaEventRecord(indexed));advanceIndexed<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,180,.25,2,false,leaders);'
 assert launch in harness;harness=harness.replace(launch,replacement)
 harness=harness.replace('if(k>=3)times.push_back(ms);','if(k>=3){times.push_back(ms);check(cudaEventElapsedTime(&ms,start,indexed));indexTimes.push_back(ms);check(cudaEventElapsedTime(&ms,indexed,end));motionTimes.push_back(ms);}')
 harness=harness.replace('// Input a is immutable:', 'std::ofstream costs(std::string(argv[3])+".costs.csv");costs<<"index_ms,motion_ms,total_ms\\n";for(size_t k=0;k<times.size();++k)costs<<indexTimes[k]<<","<<motionTimes[k]<<","<<times[k]<<"\\n";check(cudaFree(orderDevice));check(cudaFree(leaders));\n // Input a is immutable:')
 (ART/'indexed_harness.cu').write_text(harness)
 cfg=json.loads((OLD/'manifest.json').read_text());cfg.update(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),production_sha256=sha(ROOT/'src/simulator/junction_simulator.cu'),index_sha256=sha(ROOT/'tools/leader_index.cuh'),scope='Road-only immutable snapshots; index rebuilt every step; receiving and connector scans unchanged; index requires distinct positions on each lane (validated fixtures).')
 dump(ART/'manifest.json',cfg)
 for name in ('model.txt','snapshots'):
  if (OLD/name).is_dir():shutil.copytree(OLD/name,ART/name)
  else:shutil.copy2(OLD/name,ART/name)
 shutil.copy2(ROOT/'tools/snapshot_benchmark.cu',ART/'baseline_harness.cu')
 for variant,source in [('baseline',ROOT/'tools/snapshot_benchmark.cu'),('indexed',ART/'indexed_harness.cu')]:
  cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_87','-std=c++17','--ptxas-options=-v',str(source),'-o',str(ART/variant)]
  with (ART/f'{variant}-build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
 dump(ART/'binaries.json',{v:sha(ART/v) for v in ('baseline','indexed')})
def run():
 cfg=json.loads((ART/'manifest.json').read_text());result=[]
 for case in cfg['cases']:
  lanes={}
  for line in (ART/'snapshots'/f"{case['name']}.txt").read_text().splitlines()[1:]:
   ident,state,edge,pos,speed,mid=map(float,line.split())
   if int(state)==1:lanes.setdefault(int(edge),[]).append(pos)
  for positions in lanes.values():
   positions.sort();assert all(b-a>=6.5 for a,b in zip(positions,positions[1:])), 'Index fixtures require valid distinct lane positions'
 for repeat in range(3):
  for case in cfg['cases'][::1 if repeat%2==0 else -1]:
   name=case['name'];outputs=[]
   for variant in (('baseline','indexed') if repeat%2==0 else ('indexed','baseline')):
    folder=ART/'runs'/name/variant;folder.mkdir(parents=True,exist_ok=True);output=folder/f'{repeat}.csv'
    subprocess.run([str(ART/variant),str(ART/'model.txt'),str(ART/'snapshots'/f'{name}.txt'),str(output),'20'],check=True)
    outputs.append(output)
    assert sha(output)==sha(OLD/'runs'/name/'0.csv'),f'Physical mismatch {name} {variant}'
    result.append(dict(case=name,variant=variant,repeat=repeat,median_ms=statistics.median(json.loads(Path(str(output)+'.timing.json').read_text())['milliseconds'])))
   assert sha(outputs[0])==sha(outputs[1]);print(name,repeat,'exact match',flush=True)
 dump(ART/'timings.json',result)
 dump(ART/'validation.json',dict(passed=True,comparison='All 90 outputs byte-identical to frozen Milestone 2A physical outputs; all 15 fixtures, both variants, three repeats.'))
def profile():
 from profile_junction_experiment import METRICS
 for name in ('motion_50','padding_pending_8192','order_shuffle'):
  for variant in ('baseline','indexed'):
   folder=ART/'profiles'/name/variant;folder.mkdir(parents=True,exist_ok=True)
   cmd=['sudo','-n','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS+['launch__registers_per_thread','smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum']), '--kernel-name','regex:advance.*','--cache-control','none','--clock-control','none','--launch-skip','3','--launch-count','1','--force-overwrite','--export',str(folder/'motion'),str(ART/variant),str(ART/'model.txt'),str(ART/'snapshots'/f'{name}.txt'),str(folder/'output.csv'),'1']
   dump(folder/'command.json',cmd)
   with (folder/'ncu.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
   with (folder/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'motion.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
   assert sha(folder/'output.csv')==sha(OLD/'runs'/name/'0.csv')
   print('profile',name,variant,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','profile']);a=p.parse_args();globals()[a.action]()
