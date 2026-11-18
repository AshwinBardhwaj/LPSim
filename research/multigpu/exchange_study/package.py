from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/exchange-study-20261029';REPORT=ROOT/'reports/exchange-study-20261029'
reports=[p for p in REPORT.iterdir() if p.is_file() and p.suffix!='.zip']
with zipfile.ZipFile(REPORT/'report-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in reports:z.write(p,p.name)
files=reports[:]
for d in [ART/'inputs',ART/'cloud',ROOT/'artifacts/multigpu-preparation-20261027/source',ROOT/'research/multigpu/exchange_study',ROOT/'research/multigpu/cost_policies']:
    files.extend(p for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.extend(p for p in ART.iterdir() if p.is_file())
files.extend([ROOT/'research/multigpu/local_scaling/local.cu',ROOT/'research/multigpu/a100_study/concurrent.cu',ROOT/'research/multigpu/container-lock.json',ART/'local/checks.json'])
files.extend(p for p in (ART/'attempt1').rglob('*') if p.is_file())
files=sorted(set(files))
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
with zipfile.ZipFile(REPORT/'replay-bundle.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in files:z.write(p,str(p.relative_to(ROOT)))
    z.writestr('SHA256.json',json.dumps(manifest,indent=2))
for name in ['report-bundle.zip','replay-bundle.zip']:
    p=REPORT/name
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
    print(name,p.stat().st_size,hashlib.sha256(p.read_bytes()).hexdigest())
