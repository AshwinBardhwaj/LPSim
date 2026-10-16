#!/usr/bin/env python3
import os,json,subprocess,hashlib,csv
from pathlib import Path
from profile_junction_experiment import METRICS
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-junction-index-20261016';B=R/'artifacts/current/berkeley-optimized-state-20261016';N=R/'data/networks/berkeley_junction_study'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(cmd,d,env=None):
 d.mkdir(parents=True,exist_ok=True);(d/'command.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in (env or os.environ).items() if k.startswith('LPSIM_')}),indent=2))
 with (d/'run.log').open('w') as f:subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def main():
 env=dict(os.environ,LPSIM_LEADER_INDEX='1',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1')
 for scenario in sorted((R/'artifacts/junction_platoons_20261007').glob('*/case.json')):
  for exclusive in (0,1):
   for variant in (0,1):
    d=A/'validation'/scenario.parent.name/str(exclusive)/str(variant)
    if not (d/'metrics.json').exists():run([str(A/'simulator'),str(scenario.parent/'scenario.txt')],d,dict(env,LPSIM_JUNCTION_REQUEST_INDEX=str(variant),LPSIM_JUNCTION_EXCLUSIVE=str(exclusive)))
    m=json.loads((d/'metrics.json').read_text());assert all(m[k]==0 for k in ['gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'])
    if variant:
     for name in ['states.csv','events.csv','timeseries.csv','signals.csv']:assert sha(d/name)==sha(d.parent/'0'/name),(scenario,exclusive,name)
  print('small scenario exact in following/exclusive modes',scenario.parent.name,flush=True)
 cases=json.loads((B/'prepared.json').read_text())['cases']
 for c in cases:
  for t in (60,180,450):
   d=A/'local'/c['name']/str(t);out=d/'timings.csv'
   if not (d/'verified.json').exists():
    run([str(A/'local_replay'),str(N/'model.txt'),str(B/'runs'/c['name']/'0'/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(out),str(c['mode'])],d,env)
    assert len(list(csv.DictReader(out.open())))==190
    (d/'verified.json').write_text('{"priority_and_physical_summaries_exact":true}')
   print('local replay exact',c['name'],t,flush=True)
 for repeat in (1,2,3):
  for c in cases[::1 if repeat%2 else -1]:
   d=A/'runs'/c['name']/str(repeat)
   if (d/'verified.json').exists():continue
   run([str(A/'simulator'),str(B/'cases'/c['name']/'scenario.txt')],d,dict(env,LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_TRAJECTORY_STRIDE='120'))
   for name in ['states.csv','events.csv','timeseries.csv']:assert sha(d/name)==sha(B/'runs'/c['name']/'0'/name),(c,repeat,name)
   m=json.loads((d/'metrics.json').read_text());assert all(m[k]==0 for k in ['gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples']);(d/'verified.json').write_text('{"passed":true}')
   print('full run exact',c['name'],repeat,flush=True)
 metrics=METRICS+['smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','launch__registers_per_thread']
 for label in ['distributed','concentrated']:
  case=label+'_2048_seed1401_mode0'
  for t in (60,180,450):
   for variant in (0,1):
    d=A/'profiles'/case/str(t)/str(variant);d.mkdir(parents=True,exist_ok=True);lines=(B/'cases'/case/'scenario.txt').read_text().splitlines();parts=lines[1].split();parts[1]='.25';lines[1]=' '.join(parts);scenario=d/'scenario.txt';scenario.write_text('\n'.join(lines)+'\n')
    base=['sudo','-n','env','LPSIM_LEADER_INDEX=1','LPSIM_MOTION_CONTEXT=3','LPSIM_JUNCTION_PARALLEL=1',f'LPSIM_JUNCTION_REQUEST_INDEX={variant}',f'LPSIM_RESTART={B}/runs/{case}/0/snapshot_{t}.bin','/usr/local/cuda/bin/ncu','--cache-control','all','--clock-control','none','--force-overwrite']
    if not (d/'metrics.csv').exists():
     run(base+['--metrics',','.join(metrics),'--export',str(d/'pipeline'),str(A/'simulator'),str(scenario)],d)
     with (d/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'pipeline.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
    if t==180 and not (d/'source.csv').exists():
     run(base+['--kernel-name','regex:arbitrateIndexedJunctions' if variant else 'regex:arbitrateJunctions','--section','SourceCounters','--import-source','yes','--export',str(d/'source'),str(A/'simulator'),str(scenario)],d/'source-run')
     with (d/'source.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'source.ncu-rep'),'--page','source','--print-source','cuda,sass','--csv'],stdout=f,check=True)
    print('profile complete',case,t,variant,flush=True)
 # Memory check runs after all timings and profiling.
 case='distributed_2048_seed1401_mode0';d=A/'sanitizer'
 run(['sudo','-n','env','LPSIM_MOTION_CONTEXT=3','/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(A/'local_replay'),str(N/'model.txt'),str(B/'runs'/case/'0/snapshot_180.bin'),str(N/'regions.txt'),str(d/'timings.csv'),'0'],d)
 (A/'completed.json').write_text('{"passed":true}')
if __name__=='__main__':
 settings=A/'original-clocks.conf';meta={}
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 assert not settings.exists();meta['before']=clock('--show');meta['power_mode']=subprocess.check_output(['sudo','-n','nvpmodel','-q'],text=True);clock('--store',str(settings))
 try:clock();meta['fixed']=clock('--show');main()
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');(A/'clock-environment.json').write_text(json.dumps(meta,indent=2))
