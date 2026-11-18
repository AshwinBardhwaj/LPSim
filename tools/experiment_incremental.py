from pathlib import Path
import argparse,json,os,subprocess,time,hashlib,random
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validate','timing'],required=True);p.add_argument('--output',type=Path,default=r/'artifacts/current/incremental-regional-20261023');args=p.parse_args();a=args.output.resolve();b=r/'artifacts/current/berkeley-block-imbalance-20261019';cases=json.loads((b/'prepared.json').read_text())['cases'];clock=a/f'incremental-{args.phase}-clocks.conf';assert not clock.exists()
e={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};e.update(LPSIM_COMPACT_MOTION='1',LPSIM_LEADER_INDEX='1',LPSIM_LEADER_SORT='radix',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1',LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_TRAJECTORY_STRIDE='120',LPSIM_EXPERIMENT='baseline')
rows=[];policies=['bulk','incremental','periodic8']
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True)
 for batch in ([-1] if args.phase=='validate' else [-1,0,1,2]):
  jobs=[(c['name'],v) for c in cases for v in policies]
  if args.phase=='timing':random.Random(batch+1901).shuffle(jobs)
  for case,policy in jobs:
   d=a/('index-validation' if args.phase=='validate' else 'index-timing')/str(batch)/case/policy;d.mkdir(parents=True,exist_ok=True);env=e|{'LPSIM_INDEX_POLICY':policy}
   if args.phase=='validate':env['LPSIM_SAVE_SNAPSHOTS']='1'
   else:env['LPSIM_TIMING_ONLY']='1'
   cmd=[str(a/'incremental_simulator'),str(b/'cases'/case/'scenario.txt')];(d/'command.json').write_text(json.dumps(dict(command=cmd,env={k:v for k,v in env.items() if k.startswith('LPSIM_')}),indent=2));start=time.monotonic()
   with (d/'run.log').open('w') as log:q=subprocess.run(cmd,cwd=d,env=env,stdout=log,stderr=subprocess.STDOUT)
   row=dict(case=case,policy=policy,batch=batch,process_s=time.monotonic()-start,exit_code=q.returncode)
   if q.returncode:raise RuntimeError(f'{case} {policy} failed: '+(d/'run.log').read_text()[-1000:])
   row['metrics']=json.loads((d/'metrics.json').read_text())
   if args.phase=='validate':
    names=['states.csv','events.csv','signals.csv','timeseries.csv']+[x.name for x in d.glob('snapshot_*.bin')];row['exact']=all(digest(d/x)==digest(b/'runs'/case/x) for x in names)
   else:row['exact']=digest(d/'final.bin')==digest(a/'index-validation'/'-1'/case/policy/'final.bin')
   assert row['exact'];rows.append(row);(a/f'index-{args.phase}.json').write_text(json.dumps(rows,indent=2));print(batch,case,policy,'exact',row['exact'],'seconds',row['metrics']['calculation_seconds'],flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
