"""Separate full-run validation/trace generation from uninstrumented timings."""
from pathlib import Path
import argparse,json,os,subprocess,time,struct,hashlib,random
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--phase',choices=['validate','timing'],required=True);p.add_argument('--output',type=Path,default=r/'artifacts/current/gating-multirate-20261022');args=p.parse_args();a=args.output.resolve();b=r/'artifacts/current/berkeley-block-imbalance-20261019';cases=json.loads((b/'prepared.json').read_text())['cases'];clock=a/(args.phase+'-clocks.conf');assert not clock.exists()
env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_COMPACT_MOTION='1',LPSIM_LEADER_INDEX='1',LPSIM_LEADER_SORT='radix',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1',LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_TRAJECTORY_STRIDE='120')
rows=[]
def run(case,variant,d,timing=False):
 d.mkdir(parents=True,exist_ok=True);e=env|{'LPSIM_EXPERIMENT':variant}
 if timing:e['LPSIM_TIMING_ONLY']='1'
 else:e['LPSIM_SAVE_SNAPSHOTS']='1'
 if variant=='oracle':e['LPSIM_ORACLE_MASK']=str(a/'masks'/f'{case}.bin')
 cmd=[str(a/'simulator'),str(b/'cases'/case/'scenario.txt')];(d/'command.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in e.items() if k.startswith('LPSIM_')}),indent=2));start=time.monotonic()
 with (d/'run.log').open('w') as log:q=subprocess.run(cmd,cwd=d,env=e,stdout=log,stderr=subprocess.STDOUT)
 return dict(case=case,variant=variant,elapsed_s=time.monotonic()-start,exit_code=q.returncode,metrics=json.loads((d/'metrics.json').read_text()),experiment=json.loads((d/'experiment.json').read_text()))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True)
 if args.phase=='validate':
  (a/'masks').mkdir(exist_ok=True)
  for c in cases:
   case=c['name'];base=a/'validation'/case/'trace';row=run(case,'trace',base)
   names=['states.csv','events.csv','signals.csv','timeseries.csv']+[x.name for x in base.glob('snapshot_*.bin')]
   row['exact_reference']=all(digest(base/x)==digest(b/'runs'/case/x) for x in names);rows.append(row);assert row['exact_reference'] and row['exit_code']==0
   start=time.monotonic();data=(base/'request_trace.bin').read_bytes();offset=0;lists=[]
   while offset<len(data):
    count=struct.unpack_from('i',data,offset)[0];offset+=4;ids=struct.unpack_from(f'{count}i',data,offset);offset+=count*4;lists.append(ids)
   assert len(lists)==2400
   unions=[sorted(set(i for ids in lists[k:k+8] for i in ids)) for k in range(0,len(lists),8)]
   with (a/'masks'/f'{case}.bin').open('wb') as f:
    f.write(struct.pack('i',len(unions)))
    for ids in unions:f.write(struct.pack('i',len(ids)));f.write(struct.pack(f'{len(ids)}i',*ids))
   (a/'masks'/f'{case}.json').write_text(json.dumps(dict(preparation_ms=(time.monotonic()-start)*1000,trace_run_seconds=row['elapsed_s'],actual_requests=sum(map(len,lists)),candidate_updates=sum(map(len,unions))*8,total_requested_updates=c['n']*2400,windows=len(unions)),indent=2))
   for variant in ['current','oracle','multi2','multi4']:
    d=a/'validation'/case/variant;row=run(case,variant,d);row['exact_reference']=all(digest(d/x)==digest(base/x) for x in names+['final.bin']);rows.append(row)
    if variant in ['current','oracle']:assert row['exact_reference'] and row['exit_code']==0
    print(case,variant,'exact',row['exact_reference'],'invariants',row['exit_code']==0,flush=True)
   (a/'validation.json').write_text(json.dumps(rows,indent=2))
 else:
  # One full-horizon warmup and three independent shuffled measurement batches.
  for batch in [-1,0,1,2]:
   jobs=[(c['name'],v) for c in cases for v in ['baseline','current','oracle','multi2','multi4']];random.Random(1901+batch).shuffle(jobs)
   for case,variant in jobs:
    d=a/'timing'/str(batch)/case/variant;row=run(case,variant,d,True);row['batch']=batch;assert row['exit_code']==0
    target=a/'validation'/case/('trace' if variant=='baseline' else variant)/'final.bin';row['exact_validation_final']=digest(d/'final.bin')==digest(target);assert row['exact_validation_final']
    rows.append(row);(a/'timing.json').write_text(json.dumps(rows,indent=2));print(batch,case,variant,round(row['metrics']['calculation_seconds'],3),flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
