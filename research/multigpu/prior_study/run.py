from pathlib import Path
import subprocess,sys,json,time,random,hashlib,os
out=Path(sys.argv[1]);inputs=Path('/workspace/inputs');maps=out/'maps';records=[];start=time.monotonic();cases=json.loads((inputs/'cases.json').read_text());policies=['geography','paper_balanced','boundary_aware']
# CPU preprocessing executes on the actual container and is charged, including features.
subprocess.run([sys.executable,'/workspace/study/maps.py',str(inputs),str(maps)],check=True,timeout=300)
def run(c,p,repeat,steps=2400,verify=0,diag=0):
 if time.monotonic()-start>1770:raise RuntimeError('Budget deadline reached; retain partial results')
 name=f'{c["name"]}__{p}__r{repeat}v{verify}d{diag}';d=out/name;d.mkdir()
 cmd=['/workspace/bin/diagnostic' if diag else '/workspace/bin/local',str(inputs/'model.txt'),str(inputs/c['name']/'snapshot.bin'),str(maps/c['name']/(p+'.txt')),str(steps),str(d),'3',str(verify),str(diag)]
 t=time.monotonic()
 with (d/'run.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=90,env={**os.environ,'LPSIM_EXCHANGE_MODE':'3'})
 row=dict(**c,policy=p,repeat=repeat,steps=steps,verify=verify,diagnostic=diag,path=name,wall_s=time.monotonic()-t,exit_code=r.returncode);records.append(row)
 (out/'runs.json').write_text(json.dumps(records,indent=2));print(name,r.returncode,round(row['wall_s'],2),flush=True)
 if r.returncode:raise RuntimeError(name+' failed: '+(d/'run.log').read_text()[-1000:])
 row['hash']=hashlib.sha256((d/'final.bin').read_bytes()).hexdigest();(out/'runs.json').write_text(json.dumps(records,indent=2))
 return row
# Original dense reference gates from pending states, exercise all policies.
for p in policies:run(cases[0],p,-2,steps=120,verify=1)
# Independent warmup, excluded.
run(cases[0],'geography',-1,steps=120)
rng=random.Random(3300);jobs=[(c,p) for c in cases for p in policies]
for repeat in range(3):
 rng.shuffle(jobs)
 for c,p in jobs:run(c,p,repeat)
# Match every final physical byte across policies and independent repeats.
for c in cases:
 hs={r['hash'] for r in records if r['name']==c['name'] and r['steps']==2400}
 if len(hs)!=1:raise RuntimeError('Physical policy mismatch: '+c['name'])
# Diagnostics never enter timing aggregates. Collect populations at 30s and
# CUDA events for four steps per 30s window; physical/ghost checks at samples.
for c,p in jobs:
 row=run(c,p,3,diag=1)
 reference=next(r['hash'] for r in records if r['name']==c['name'] and r['repeat']==0)
 if row['hash']!=reference:raise RuntimeError('Diagnostic physical mismatch')
# Retain one final payload/case; all hashes are retained.
for r in records:
 if r['repeat']!=0 or r['policy']!='geography':(out/r['path']/'final.bin').unlink()
(out/'checks.json').write_text(json.dumps(dict(exact_final_equivalence=True,reference_steps=360,sampled_physical_checks=48*20,elapsed_s=time.monotonic()-start),indent=2))
