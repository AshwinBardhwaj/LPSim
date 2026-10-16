from pathlib import Path
import os,json,subprocess,csv,hashlib
from profile_junction_experiment import METRICS
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-motion-context-20261014';B=R/'artifacts/current/berkeley-state-study-20261014';N=R/'data/networks/berkeley_junction_study'
def run(cmd,folder,env=None):
 folder.mkdir(parents=True,exist_ok=True)
 (folder/'command.json').write_text(json.dumps(cmd))
 with (folder/'run.log').open('w') as f:subprocess.run(cmd,cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cases=json.loads((B/'prepared.json').read_text())['cases']
 for case in cases:
  for t in (60,180,450):
   for flag in (0,1,2,3):
    d=A/'replays'/case['name']/str(t)/str(flag);out=d/'output.json'
    env=dict(os.environ,LPSIM_MOTION_CONTEXT=str(flag),LPSIM_STUDY_MODE=str(case['mode']))
    if not out.exists():run([str(A/'replay'),str(N/'model.txt'),str(B/'runs'/case['name']/'0'/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(out),'-1','-1'],d,env)
    assert json.loads(out.read_text())['physical_digest']==json.loads((B/'replays'/case['name']/str(t)/'selector_-1.json').read_text())['physical_digest'],(case,t,flag)
   print('snapshot',case['name'],t,'all variants exact',flush=True)
 for scenario in sorted((R/'artifacts/junction_platoons_20261007').glob('*/case.json')):
  for flag in (1,2,3):
   d=A/'validation'/scenario.parent.name/str(flag)
   env=dict(os.environ,LPSIM_MOTION_CONTEXT=str(flag),LPSIM_LEADER_INDEX='1',LPSIM_JUNCTION_PARALLEL='1')
   run([str(A/'simulator'),str(scenario.parent/'scenario.txt')],d,env)
   for name in ['states.csv','events.csv','timeseries.csv','signals.csv']:assert sha(d/name)==sha(B/'validation'/scenario.parent.name/'reference'/name),(scenario,flag,name)
  print('small scenario',scenario.parent.name,'all exact',flush=True)
 # Complete trajectories, both isolated changes and their combination.
 for case in cases:
  for flag in (1,2,3):
   d=A/'runs'/case['name']/str(flag)
   if not (d/'verified.json').exists():
    env=dict(os.environ,LPSIM_MOTION_CONTEXT=str(flag),LPSIM_LEADER_INDEX='1',LPSIM_JUNCTION_PARALLEL='1',LPSIM_TRAJECTORY_STRIDE='120')
    run([str(A/'simulator'),str(B/'cases'/case['name']/'scenario.txt')],d,env)
    for name in ['states.csv','events.csv','timeseries.csv']:assert sha(d/name)==sha(B/'runs'/case['name']/'0'/name),(case,flag,name)
    m=json.loads((d/'metrics.json').read_text());assert all(m[k]==0 for k in ['gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'])
    (d/'verified.json').write_text('{"exact_trajectory":true}')
   print('trajectory',case['name'],flag,'exact',flush=True)
 for label in ['distributed','concentrated']:
  case=label+'_2048_seed1401_mode0';snap=B/'runs'/case/'0/snapshot_180.bin'
  for flag in (0,1,2,3):
   d=A/'profiles'/case/str(flag);d.mkdir(parents=True,exist_ok=True)
   base=['sudo','-n','env',f'LPSIM_MOTION_CONTEXT={flag}','LPSIM_STUDY_MODE=0','/usr/local/cuda/bin/ncu','--kernel-name','regex:advance','--launch-skip','4','--launch-count','1','--clock-control','none','--cache-control','none','--force-overwrite']
   args=[str(A/'replay'),str(N/'model.txt'),str(snap),str(N/'regions.txt'),str(d/'output.json'),'-1','-1']
   for kind,options in [('metrics',['--metrics',','.join(METRICS)]),('source',['--section','SourceCounters','--import-source','yes'])]:
    if (d/(kind+'.csv')).exists():continue
    run(base+options+['--export',str(d/kind)]+args,d/kind)
    cmd=['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/(kind+'.ncu-rep')),'--page','raw' if kind=='metrics' else 'source','--csv']
    if kind=='source':cmd+=['--print-source','cuda,sass']
    with (d/(kind+'.csv')).open('w') as f:subprocess.run(cmd,stdout=f,check=True)
   assert json.loads((d/'output.json').read_text())['physical_digest']==json.loads((B/'replays'/case/'180/selector_-1.json').read_text())['physical_digest']
   print('profile',case,flag,flush=True)
if __name__=='__main__':
 settings=A/'original-clocks.conf';meta={}
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 if settings.exists():settings.unlink()
 meta['before']=clock('--show');clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show');main()
 finally:
  meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');(A/'clock-environment.json').write_text(json.dumps(meta,indent=2))
