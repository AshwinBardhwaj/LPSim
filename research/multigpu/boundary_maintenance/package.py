from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3];art=ROOT/'artifacts/boundary-maintenance-20261103';report=ROOT/'reports/boundary-maintenance-20261103'
files=[p for p in report.rglob('*') if p.is_file() and p.suffix!='.zip']
with zipfile.ZipFile(report/'report-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in files:z.write(p,'report/'+str(p.relative_to(report)))
files+= [p for p in (ROOT/'research/multigpu/boundary_maintenance').rglob('*') if p.is_file() and '__pycache__' not in str(p)]
files+= [p for p in (art/'inputs').rglob('*') if p.is_file()]
files+= [p for p in (art/'cloud').glob('*.tar.gz')]
files+= [p for p in art.glob('*.json')]
files+= [p for p in (ROOT/'artifacts/multigpu-preparation-20261027/source').rglob('*') if p.is_file()]
files+= [ROOT/'research/multigpu/exchange_study/local.cu',ROOT/'research/multigpu/exchange_study/trace.cuh',ROOT/'research/multigpu/container-lock.json']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(report/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2))
with zipfile.ZipFile(report/'replay-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 for p in files:z.write(p,str(p.relative_to(ROOT)))
 z.write(report/'bundle-manifest.json','bundle-manifest.json')
print('Packaged',len(files),'files')
