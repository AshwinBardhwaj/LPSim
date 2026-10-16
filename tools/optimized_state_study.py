#!/usr/bin/env python3
"""Frozen optimized state-cost rerun; outputs isolated from earlier milestones."""
import os,json,subprocess,shutil,csv,hashlib,sys
from pathlib import Path
import berkeley_state_study as base
from profile_junction_experiment import METRICS
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-optimized-state-20261016';B=R/'artifacts/current/berkeley-state-study-20261014';PRE=R/'artifacts/current/berkeley-motion-context-20261014';N=base.NET
os.environ['LPSIM_MOTION_CONTEXT']='3'
def dump(p,x):base.dump(p,x)
def run(cmd,d,env=None):
 d.mkdir(parents=True,exist_ok=True);dump(d/'command.json',dict(command=cmd,environment={k:v for k,v in (env or os.environ).items() if k.startswith('LPSIM_')}))
 with (d/'run.log').open('w') as f:subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def prepare():
 A.mkdir(parents=True,exist_ok=True)
 for name in ['prepared.json','validation.json']:shutil.copy2(B/name,A/name)
 shutil.copytree(B/'cases',A/'cases',dirs_exist_ok=True)
 for name in ['simulator','replay']:shutil.copy2(PRE/name,A/name)
 sources=['src/simulator/scalable_junction_simulator.cu','src/simulator/junction_simulator.h','tools/leader_index.cuh','tools/motion_context.cuh','tools/berkeley_replay.cu','tools/optimized_state_study.py']
 for f in sources:
  dest=A/'source'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(R/f,dest)
 dump(A/'freeze.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_hashes={f:base.sha(R/f) for f in sources},binary_hashes={f:base.sha(A/f) for f in ['simulator','replay']},motion_context=3,timestep=.25,horizon=600,block=128,seed_holdout=1402,region_holdout=[12,13,14,15],ridge=.01,flags='-O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17'))
 source=(R/'src/simulator/scalable_junction_simulator.cu').read_text()
 for name,path in [('junction_simulator.h',R/'src/simulator/junction_simulator.h'),('../../tools/leader_index.cuh',R/'tools/leader_index.cuh'),('../../tools/motion_context.cuh',R/'tools/motion_context.cuh')]:source=source.replace('"'+name+'"','"'+str(path)+'"')
 source=source.replace('namespace {','namespace {\n__device__ unsigned long long* studyCounts;',1)
 a=source.index('__global__ void advance(');b=source.index('// Deterministic junction arbitration')
 kernel=source[a:b].replace('Car c=old[i];','unsigned long long scans[2]={};auto begin=clock64();Car c=old[i];',1).replace('if(open)for(int j=0;j<n;++j){','if(open)for(int j=0;j<n;++j){scans[0]++;').replace('j=connectorHead?connectorNext[j]:j+1){','j=connectorHead?connectorNext[j]:j+1){scans[1]++;').replace('next[i]=c;','next[i]=c;studyCounts[i*3]=scans[0];studyCounts[i*3+1]=scans[1];studyCounts[i*3+2]=clock64()-begin;')
 (A/'instrumented.cu').write_text(source[:a]+kernel+source[b:]);(A/'instrumented_replay.cu').write_text((R/'tools/berkeley_replay.cu').read_text().replace('../src/simulator/scalable_junction_simulator.cu',str(A/'instrumented.cu')))
 run(['/usr/local/cuda/bin/nvcc','-O3','--use_fast_math','-lineinfo','-arch=sm_87','-std=c++17','-DSTUDY_INSTRUMENT',str(A/'instrumented_replay.cu'),'-o',str(A/'instrumented_replay')],A/'build-instrumented')
def replays():
 cases=json.loads((A/'prepared.json').read_text())['cases']
 for case in cases:
  for t in (60,180,450):
   d=A/'replays'/case['name']/str(t);d.mkdir(parents=True,exist_ok=True);snap=B/'runs'/case['name']/'0'/f'snapshot_{t}.bin';env=dict(os.environ,LPSIM_STUDY_MODE=str(case['mode']))
   for selector in range(-1,22):
    out=d/f'selector_{selector}.json'
    if not out.exists():run([str(A/'replay'),str(N/'model.txt'),str(snap),str(N/'regions.txt'),str(out),str(selector),'-1'],d/f'command_{selector}',env)
    assert json.loads(out.read_text())['physical_digest']==json.loads((B/'replays'/case['name']/str(t)/out.name).read_text())['physical_digest']
   for speed in (0,5):
    out=d/f'speed_{speed}.json'
    if not out.exists():run([str(A/'replay'),str(N/'model.txt'),str(snap),str(N/'regions.txt'),str(out),'-1',str(speed)],d/f'command_speed_{speed}',env)
    assert json.loads(out.read_text())['physical_digest']==json.loads((B/'replays'/case['name']/str(t)/out.name).read_text())['physical_digest']
   for control in (B/'replays'/case['name']/str(t)).glob('control_*.json'):
    label=control.stem[8:];out=d/control.name
    settings={'shuffle':{'LPSIM_REPLAY_ORDER':'shuffle'},'lane_order':{'LPSIM_REPLAY_ORDER':'lane'},'padding':{'LPSIM_PADDING':str(case['n'])},'half_active':{'LPSIM_HALF_ACTIVE':'1'},'block64':{'LPSIM_REPLAY_BLOCK':'64'},'block256':{'LPSIM_REPLAY_BLOCK':'256'}}
    fixture=control.parent/(label+'.bin') if label in ('mix_all','mix_half') else snap
    if not out.exists():run([str(A/'replay'),str(N/'model.txt'),str(fixture),str(N/'regions.txt'),str(out),'-1','-1'],d/f'command_{label}',dict(env,**settings.get(label,{})))
    assert json.loads(out.read_text())['physical_digest']==json.loads(control.read_text())['physical_digest']
   out=d/'instrumented.json'
   if not out.exists():run([str(A/'instrumented_replay'),str(N/'model.txt'),str(snap),str(N/'regions.txt'),str(out),'-1','-1'],d/'command_instrumented',env)
   assert json.loads(out.read_text())['physical_digest']==json.loads((d/'selector_-1.json').read_text())['physical_digest']
   print('controlled + regional/state replays',case['name'],t,flush=True)
def fullruns():
 base.ART=A;base.run()
 for c in json.loads((A/'prepared.json').read_text())['cases']:
  for rep in range(4):
   for f in ('states.csv','events.csv','timeseries.csv'):assert base.sha(A/'runs'/c['name']/str(rep)/f)==base.sha(B/'runs'/c['name']/'0'/f)
 print('All full trajectories match frozen baseline',flush=True)
def profiles():
 metrics=METRICS+['smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','launch__registers_per_thread']
 for c in json.loads((A/'prepared.json').read_text())['cases']:
  if c['seed']!=1401:continue
  for t in (60,180,450):
   d=A/'profiles'/c['name']/str(t);d.mkdir(parents=True,exist_ok=True);lines=(A/'cases'/c['name']/'scenario.txt').read_text().splitlines();parts=lines[1].split();parts[1]='.25';lines[1]=' '.join(parts);scenario=d/'scenario.txt';scenario.write_text('\n'.join(lines)+'\n')
   envargs=['LPSIM_MOTION_CONTEXT=3','LPSIM_LEADER_INDEX=1','LPSIM_JUNCTION_PARALLEL=1',f'LPSIM_RESTART={A}/runs/{c["name"]}/0/snapshot_{t}.bin']
   if not (d/'metrics.csv').exists():
    # No kernel filter: capture every kernel in the single complete step, including Thrust sorting.
    cmd=['sudo','-n','env',*envargs,'/usr/local/cuda/bin/ncu','--metrics',','.join(metrics),'--cache-control','all','--clock-control','none','--force-overwrite','--export',str(d/'pipeline'),str(A/'simulator'),str(scenario)]
    run(cmd,d)
    with (d/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'pipeline.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
   # Match physical step counters with the uninterrupted trajectory.
   keys=['requested','pending','not_departed','road_stopped','road_moving','connector','completed','entries','spawned','completed_step','blocked_space','blocked_conflict_or_yield']
   actual=list(csv.DictReader((d/'step_metrics.csv').open()))[0];expected=next(r for r in csv.DictReader((A/'runs'/c['name']/'0/step_metrics.csv').open()) if float(r['time'])==t)
   assert all(actual[k]==expected[k] for k in keys)
   print('complete pipeline profile',c['name'],t,flush=True)
   if t==180 and c['mode']==0 and c['n']==2048:
    for selector in range(16,22):
     sd=d/f'state_{selector}';sd.mkdir(exist_ok=True)
     if (sd/'metrics.csv').exists():continue
     cmd=['sudo','-n','env','LPSIM_MOTION_CONTEXT=3','LPSIM_STUDY_MODE=0','/usr/local/cuda/bin/ncu','--metrics',','.join(metrics),'--kernel-name','regex:advance','--launch-skip','4','--launch-count','1','--cache-control','all','--clock-control','none','--force-overwrite','--export',str(sd/'kernel'),str(A/'replay'),str(N/'model.txt'),str(A/'runs'/c['name']/'0'/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(sd/'output.json'),str(selector),'-1']
     run(cmd,sd)
     with (sd/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(sd/'kernel.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
    if not (d/'source.csv').exists():
     run(['sudo','-n','env',*envargs,'/usr/local/cuda/bin/ncu','--section','SourceCounters','--import-source','yes','--cache-control','all','--clock-control','none','--force-overwrite','--export',str(d/'source'),str(A/'simulator'),str(scenario)],d/'source-run')
     with (d/'source.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'source.ncu-rep'),'--page','source','--print-source','cuda,sass','--csv'],stdout=f,check=True)
def main():
 if len(sys.argv)>1 and sys.argv[1]=='prepare':prepare();return
 settings=A/'original-clocks.conf';meta={}
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 if settings.exists():raise RuntimeError('Use explicit resume after checking saved clocks')
 meta['power_mode']=subprocess.check_output(['sudo','-n','nvpmodel','-q'],text=True);meta['before']=clock('--show');clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show');replays();fullruns();profiles();dump(A/'completed.json',{'passed':True})
 finally:
  meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');dump(A/'clock-environment.json',meta)
if __name__=='__main__':main()
