from pathlib import Path
import json,shutil,hashlib
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'artifacts/boundary-maintenance-20261103/inputs';src=ROOT/'artifacts/exchange-study-20261029/inputs'
out.mkdir(exist_ok=True)
shutil.copytree(src/'models',out/'models',dirs_exist_ok=True)
cases=json.loads((src/'cases.json').read_text());cases.append(dict(name='berkeley_developed_2048',network='berkeley_junction_study',n=2048,kind='saved reference trajectory at t=180 s; includes pending, road and connector states'))
for c in cases:
 d=out/c['name'];d.mkdir(exist_ok=True);m=src/('berkeley_8192' if c['name']=='berkeley_developed_2048' else c['name'])
 snap=m/'snapshot.bin' if c['name']!='berkeley_developed_2048' else ROOT/'artifacts/multigpu-preparation-20261027/validation/reference_2048_t0-geography-bulk-mode0/snapshot_180.bin'
 shutil.copy2(snap,d/'snapshot.bin')
 for p in ['paper_balanced','boundary_aware']:shutil.copy2(m/(p+'_3.txt'),d/(p+'_3.txt'))
(out/'cases.json').write_text(json.dumps(cases,indent=2));(out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2))
print('Frozen cases:',len(cases))
