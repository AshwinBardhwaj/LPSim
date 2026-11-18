from pathlib import Path
import subprocess,os,hashlib,json
root=Path(__file__).resolve().parents[3];base=root/'artifacts/shadow-study-20261104';checks=[]
for policy in ['fixed','reactive','forecast']:
 d=base/('local-replay-'+policy);d.mkdir(exist_ok=True);owner=base/'inputs/test_mixed_16384/fixed.txt' if policy=='fixed' else base/'local-maps'/(policy+'.txt')
 cmd=['/tmp/lpsim-shadow-build/diagnostic',str(base/'inputs/model.txt'),str(base/'local-capture/snapshot_180.bin'),str(owner),'240',str(d),'3','1','1']
 p=subprocess.run(cmd,env={**os.environ,'LPSIM_LOGICAL':'1','LPSIM_EXCHANGE_MODE':'3'},capture_output=True,text=True,timeout=120);(d/'run.log').write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError((d/'run.log').read_text())
 same=(d/'final.bin').read_bytes()==(base/'local-capture/snapshot_240.bin').read_bytes()
 if not same:raise RuntimeError('Restart continuity failure '+policy)
 checks.append(dict(policy=policy,every_step_reference_checks=240,uninterrupted_final_identical=same));print(checks[-1],flush=True)
(base/'local-checks.json').write_text(json.dumps(checks,indent=2))
