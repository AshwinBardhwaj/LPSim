from pathlib import Path
import subprocess,sys,json,time,random,hashlib,shutil
out=Path(sys.argv[1]);phase=sys.argv[2];inputs=Path('/workspace/inputs');exe='/workspace/bin/concurrent';started=time.monotonic();records=[]
policies=['geography','vehicle_count','communication_graph']
def run(case,policy,backend,steps,verify=0,instrument=0,repeat=0,snapshot=None,extra=None,policy_dir=None):
 name=f'{case}__{policy}__{backend}__v{verify}i{instrument}r{repeat}'
 if extra:name+='__'+extra
 d=out/name;d.mkdir();cmd=[exe,str(inputs/'network/model.txt'),str(snapshot or inputs/case/'snapshot.bin'),str((policy_dir or inputs/case)/(policy+'.txt')),str(steps),str(d),backend,str(verify),str(instrument),str(repeat)]
 before=time.monotonic()
 with (d/'run.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=150)
 record=dict(case=case,policy=policy,backend=backend,verify=verify,instrument=instrument,repeat=repeat,path=name,wall_s=time.monotonic()-before,exit_code=p.returncode,argv=cmd)
 records.append(record);(out/'runs.json').write_text(json.dumps(records,indent=2));print(record,flush=True)
 if p.returncode:raise RuntimeError(name+' failed')
 record['final_sha256']=hashlib.sha256((d/'final.bin').read_bytes()).hexdigest()
 (out/'runs.json').write_text(json.dumps(records,indent=2))
 return d
if phase=='concurrent_gate':
 for case in ['reference_8192_t180','reference_16384_t180']:
  results=[]
  for policy in policies:
   d=run(case,policy,'peer',120,verify=1,instrument=1);results.append((d/'final.bin').read_bytes())
  if len(set(results))!=1:raise RuntimeError('Policy final mismatch')
 # Exercise initially pending demand, spawning, signals and many migrations over 600 s.
 for policy in policies:run('train_bursty_8192',policy,'peer',2400,verify=1)
elif phase.startswith('matrix'):
 n=int(phase.split('_')[1]);rng=random.Random(2710)
 for pattern in ['distributed','concentrated','bursty']:
  case=f'train_{pattern}_{n}'
  # Current-state count balancing requires an actual populated state, not an empty fallback.
  seed=run(case,'geography','single',720,verify=1,instrument=1,repeat=-1)
  snapshot=seed/'snapshot_180.bin'
  sys.path.insert(0,'/workspace/source/research/multigpu')
  from prepare import read_snapshot,partition
  geometry=json.loads((inputs/'network/geometry.json').read_text());_,cars=read_snapshot(snapshot)
  policy_dir=out/(case+'__allocation');policy_dir.mkdir();allocation=[]
  for policy in policies:
   before=time.monotonic();assignment,meta=partition(geometry,cars,policy);elapsed=time.monotonic()-before
   (policy_dir/(policy+'.txt')).write_text('\n'.join(map(str,assignment))+'\n')
   meta['construction_ms']=elapsed*1000;meta['decision_time_s']=180;allocation.append(meta)
  (policy_dir/'allocation.json').write_text(json.dumps(allocation,indent=2))
  configs=[('geography','single')]+[(p,'peer') for p in policies]
  hashes=set()
  # Warm-up is excluded. Each repetition randomizes all policy orders.
  for repeat in range(6):
   order=configs.copy();rng.shuffle(order)
   for policy,backend in order:
    d=run(case,policy,backend,1680,repeat=repeat,snapshot=snapshot,policy_dir=policy_dir);hashes.add(hashlib.sha256((d/'final.bin').read_bytes()).hexdigest())
    # Preserve every final hash, but only one identical final payload per policy.
    if repeat>0:(d/'final.bin').unlink()
   if time.monotonic()-started>760:raise RuntimeError('Budget deadline; partial results retained')
  if len(hashes)!=1:raise RuntimeError('Final physical outputs differ across runs')
  for policy,backend in configs:
   d=run(case,policy,backend,1680,instrument=1,repeat=7,snapshot=snapshot,policy_dir=policy_dir)
   if hashlib.sha256((d/'final.bin').read_bytes()).hexdigest() not in hashes:raise RuntimeError('Diagnostic physical mismatch')
   (d/'final.bin').unlink()
else:raise RuntimeError('Unknown phase '+phase)
