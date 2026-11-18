from pathlib import Path
import subprocess,sys,json,time,random,hashlib,os,shutil,csv
from propose import make
out=Path(sys.argv[1]);inputs=Path('/workspace/inputs');cases=json.loads((inputs/'cases.json').read_text());records=[];proposals=[];started=time.monotonic();policies=['fixed','reactive','forecast']
def run(case,snapshot,owner,steps,tag,verify=0,diag=0,capture=False,repeat=-1,policy='fixed',decision=0):
 if time.monotonic()-started>850:raise RuntimeError('Budget deadline; preserve partial outputs')
 folder=out/tag;folder.mkdir();env={**os.environ,'LPSIM_EXCHANGE_MODE':'3'}
 if capture:env['LPSIM_CAPTURE']='1'
 cmd=['/workspace/bin/diagnostic' if diag else '/workspace/bin/local',str(inputs/'model.txt'),str(snapshot),str(owner),str(steps),str(folder),'3',str(verify),str(diag)]
 begin=time.monotonic()
 with (folder/'run.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=80,env=env)
 row=dict(case=case,policy=policy,decision_s=decision,repeat=repeat,verify=verify,diagnostic=diag,capture=capture,steps=steps,path=tag,wall_ms=1000*(time.monotonic()-begin),exit_code=p.returncode);records.append(row)
 if p.returncode:
  (out/'runs.json').write_text(json.dumps(records,indent=2));raise RuntimeError(tag+': '+(folder/'run.log').read_text()[-1000:])
 row['hash']=hashlib.sha256((folder/'final.bin').read_bytes()).hexdigest();(out/'runs.json').write_text(json.dumps(records,indent=2));print(tag,round(row['wall_ms'],2),flush=True);return folder
topology_start=time.perf_counter();geometry=json.loads((inputs/'geometry.json').read_text());(out/'shared-preparation.json').write_text(json.dumps(dict(topology_load_ms=1000*(time.perf_counter()-topology_start),note='Immutable topology reused by all proposals; one-time additional shadow setup cost.'),indent=2))
jobs=[]
for c in cases:
 name=c['name'];capture=run(name,inputs/name/'snapshot.bin',inputs/name/'fixed.txt',1920,name+'__capture',diag=1,capture=True)
 for t in [180,300,420]:
  maps=out/'maps'/name/str(t);maps.mkdir(parents=True);shutil.copy2(inputs/name/'fixed.txt',maps/'fixed.txt')
  for policy in ['reactive','forecast']:proposals.append(make(inputs,capture,name,t,maps,policy,geometry=geometry))
  expected=hashlib.sha256((capture/f'snapshot_{t+60}.bin').read_bytes()).hexdigest()
  for policy in policies:jobs.append(dict(case=name,t=t,snapshot=capture/f'snapshot_{t}.bin',owner=maps/(policy+'.txt'),policy=policy,expected=expected))
(out/'proposals.json').write_text(json.dumps(proposals,indent=2))
# Every-step reference gate from an occupied snapshot for each policy, prior to timing.
for j in jobs[:3]:run(j['case'],j['snapshot'],j['owner'],40,f'gate_{j["policy"]}',verify=1,policy=j['policy'],decision=j['t'])
# Independent excluded warmup.
j=jobs[0];run(j['case'],j['snapshot'],j['owner'],40,'warmup',policy='fixed')
rng=random.Random(3404)
for repeat in range(5):
 rng.shuffle(jobs)
 for j in jobs:
  d=run(j['case'],j['snapshot'],j['owner'],240,f'{j["case"]}__t{j["t"]}__{j["policy"]}__r{repeat}',repeat=repeat,policy=j['policy'],decision=j['t'])
  if hashlib.sha256((d/'final.bin').read_bytes()).hexdigest()!=j['expected']:raise RuntimeError('Restart/candidate physical mismatch: '+str(d))
  if repeat>0:(d/'final.bin').unlink()
for j in jobs:
 d=run(j['case'],j['snapshot'],j['owner'],240,f'{j["case"]}__t{j["t"]}__{j["policy"]}__diagnostic',diag=1,policy=j['policy'],decision=j['t'])
 if hashlib.sha256((d/'final.bin').read_bytes()).hexdigest()!=j['expected']:raise RuntimeError('Diagnostic mismatch')
 (d/'final.bin').unlink()
(out/'checks.json').write_text(json.dumps(dict(restart_and_policy_final_equivalence=True,reference_steps=120,unprofiled_repeats=5,matched_windows=12,successful_runs=len(records),elapsed_s=time.monotonic()-started),indent=2))
