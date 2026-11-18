from pathlib import Path
import json,subprocess,time,hashlib,random,sys
out=Path(sys.argv[1]);inputs=Path('/workspace/inputs');start=time.monotonic();records=[]
def run(case,policy,ranks,verify,diagnostic,repeat,steps=60,dense=False):
 name=f"{case['name']}__{policy}__k{ranks}__v{verify}d{diagnostic}r{repeat}"+('__dense' if dense else '')
 folder=out/name;folder.mkdir();snapshot=inputs/case['name']/'snapshot.bin';model=inputs/'models'/case['network']/'model.txt';owners=inputs/case['name']/f'{policy}_{ranks}.txt'
 if dense:cmd=['/workspace/bin/dense',str(model),str(snapshot),str(owners),str(steps),str(folder),'single',str(verify),str(diagnostic),str(repeat)]
 else:cmd=['/workspace/bin/local',str(model),str(snapshot),str(owners),str(steps),str(folder),str(ranks),str(verify),str(diagnostic)]
 before=time.monotonic()
 with (folder/'run.log').open('w') as log:p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=100)
 record=dict(case=case['name'],network=case['network'],n=case['n'],policy=policy,ranks=ranks,verify=verify,diagnostic=diagnostic,repeat=repeat,dense=dense,steps=steps,process_seconds=time.monotonic()-before,exit_code=p.returncode,path=name)
 if p.returncode:raise RuntimeError(name+'\n'+(folder/'run.log').read_text())
 record['hash']=hashlib.sha256((folder/'final.bin').read_bytes()).hexdigest();record['metrics']=json.loads((folder/'metrics.json').read_text());records.append(record);(out/'runs.json').write_text(json.dumps(records,indent=2));print(name,round(record['process_seconds'],2),flush=True)
 if repeat>0 and not verify:(folder/'final.bin').unlink()
 return record['hash']
cases=json.loads((inputs/'cases.json').read_text())
# Historical simulated queues/connectors test the new authority/priority behavior.
historical=json.loads((inputs/'historical.json').read_text())
for k in [2,3]:
 for policy in ['paper_balanced','paper_community']:run(historical,policy,k,1,1,-2,20)
for case in cases:
 configs=[('geography',1,False),('geography',1,True)]+[(p,k,False) for k in [2,3] for p in ['geography','paper_balanced','paper_community']]
 hashes=set()
 for p,k,dense in configs:hashes.add(run(case,p,k,1,1,-1,20,dense))
 if len(hashes)!=1:raise RuntimeError('20-step physical hashes differ '+case['name'])
 hashes=set();rng=random.Random(2810)
 for repeat in range(4):
  order=configs.copy();rng.shuffle(order)
  for p,k,dense in order:
   if time.monotonic()-start>780:raise RuntimeError('Budgeted experiment deadline; preserve partial results')
   hashes.add(run(case,p,k,0,0,repeat,60,dense))
 if len(hashes)!=1:raise RuntimeError('60-step final hashes differ '+case['name'])
(out/'complete.json').write_text(json.dumps(dict(passed=True,cases=len(cases),records=len(records),seconds=time.monotonic()-start),indent=2))
