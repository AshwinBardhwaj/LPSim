from pathlib import Path
import os,subprocess,json,hashlib
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'artifacts/boundary-maintenance-20261103/local';inputs=ROOT/'artifacts/exchange-study-20261029/inputs';rows=json.loads((out/'checks.json').read_text()) if (out/'checks.json').exists() else [];expected={r['case']:r['hash'] for r in rows}
configs=[('berkeley_8192',40),('berkeley_developed_2048',80)]
for case,steps in configs:
 network='sf_partial_local_scaling' if case.startswith('sf') else 'berkeley_junction_study'
 snapshot=inputs/case/'snapshot.bin' if case!='berkeley_developed_2048' else ROOT/'artifacts/multigpu-preparation-20261027/validation/reference_2048_t0-geography-bulk-mode0/snapshot_180.bin'
 mapcase='sf_98304' if case.startswith('sf') else 'berkeley_8192'
 for policy in ['paper_balanced','boundary_aware']:
  for mode in [0,1,2]:
   dest=out/f'{case}_{policy}_m{mode}';dest.mkdir(exist_ok=True)
   if any(r['case']==case and r['policy']==policy and r['mode']==mode for r in rows):continue
   cmd=['/tmp/lpsim-boundary-maintenance',str(inputs/'models'/network/'model.txt'),str(snapshot),str(inputs/mapcase/(policy+'_3.txt')),str(steps),str(dest),'3','1','1']
   with (dest/'run.log').open('w') as log:r=subprocess.run(cmd,env=dict(os.environ,LPSIM_LOGICAL='1',LPSIM_EXCHANGE_MODE='3',LPSIM_LIST_MAINTENANCE=str(mode)),stdout=log,stderr=subprocess.STDOUT,timeout=120)
   if r.returncode:raise RuntimeError(str(dest)+' '+(dest/'run.log').read_text())
   h=hashlib.sha256((dest/'final.bin').read_bytes()).hexdigest()
   if case in expected:assert h==expected[case],dest
   expected[case]=h
   rows.append(dict(case=case,steps=steps,policy=policy,mode=mode,hash=h,passed=True));(out/'checks.json').write_text(json.dumps(rows,indent=2));print(dest.name,'PASS',flush=True)
print('Exact timesteps:',sum(x['steps'] for x in rows))
