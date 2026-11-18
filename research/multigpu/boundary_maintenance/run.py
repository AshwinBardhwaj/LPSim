from pathlib import Path
import hashlib,json,os,random,subprocess,sys,time
out=Path(sys.argv[1]);inputs=Path('/workspace/inputs');started=time.monotonic();records=[];expected={};wire={};counter=0
cases=json.loads((inputs/'cases.json').read_text())
def run(case,policy,mode,phase,repeat=0,steps=60,verify=False,diagnostic=False,frozen=False,force=False):
 global counter
 if time.monotonic()-started>420:raise RuntimeError('Matrix deadline')
 counter+=1;name=f'{counter:04d}_{case["name"]}_{policy}_m{mode}_{phase}_r{repeat}';folder=out/name;folder.mkdir()
 exe='/workspace/bin/'+('frozen' if frozen else 'diagnostic' if diagnostic else 'local')
 cmd=[exe,str(inputs/'models'/case['network']/'model.txt'),str(inputs/case['name']/'snapshot.bin'),str(inputs/case['name']/(policy+'_3.txt')),str(steps),str(folder),'3',str(int(verify)),str(int(diagnostic))]
 env=dict(os.environ,LPSIM_EXCHANGE_MODE='3',LPSIM_LIST_MAINTENANCE=str(mode))
 if force:env['LPSIM_FORCE_REBUILD']='1'
 before=time.monotonic()
 with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=min(60,max(1,440-(time.monotonic()-started))))
 if p.returncode:raise RuntimeError(name+'\n'+(folder/'run.log').read_text())
 h=hashlib.sha256((folder/'final.bin').read_bytes()).hexdigest();key=(case['name'],steps)
 if key in expected and expected[key]!=h:raise RuntimeError('Physical hash mismatch '+name)
 expected[key]=h;metrics=json.loads((folder/'metrics.json').read_text());wk=(case['name'],steps,policy)
 if wk in wire and wire[wk]!=metrics['wire_bytes']:raise RuntimeError('Wire payload changed '+name)
 wire[wk]=metrics['wire_bytes'];row=dict(metrics);row.update(case=case['name'],policy=policy,mode=mode,phase=phase,repeat=repeat,steps=steps,diagnostic=diagnostic,frozen=frozen,forced=force,path=name,final_sha256=h,process_seconds=time.monotonic()-before)
 records.append(row);(out/'runs.json').write_text(json.dumps(records,indent=2));(folder/'final.bin').unlink();print(name,round(metrics['compute_ms'],3),'ms',flush=True)
 return row
# Physical correctness and original-backend equivalence before independent timing.
for c in cases:
 run(c,'paper_balanced',0,'frozen_gate',steps=20,verify=True,frozen=True)
 for p in ['paper_balanced','boundary_aware']:
  for m in [0,1,2]:run(c,p,m,'gate',steps=20,verify=True)
# Component traces are separate from all benchmark timings, and validate every step.
for c in cases:
 for m in [0,1,2]:run(c,'paper_balanced',m,'diagnostic',steps=40,verify=True,diagnostic=True)
run(cases[-1],'paper_balanced',2,'forced_fallback',steps=40,verify=True,diagnostic=True,force=True)
for c in cases:
 configs=[(p,m) for p in ['paper_balanced','boundary_aware'] for m in [0,1,2]]
 for p,m in configs:run(c,p,m,'warmup')
 for rep in range(5):
  order=configs[:];random.Random(11030+rep).shuffle(order)
  for p,m in order:run(c,p,m,'test',repeat=rep)
# Longer window on real trajectory, same predeclared paper-balanced partition.
for m in [0,1,2]:run(cases[-1],'paper_balanced',m,'long_warmup',steps=240)
for rep in range(5):
 order=[0,1,2];random.Random(21030+rep).shuffle(order)
 for m in order:run(cases[-1],'paper_balanced',m,'long_test',repeat=rep,steps=240)
(out/'complete.json').write_text(json.dumps(dict(passed=True,records=len(records),exact_timesteps=sum(r['steps'] for r in records if r['verified']),seconds=time.monotonic()-started),indent=2))
