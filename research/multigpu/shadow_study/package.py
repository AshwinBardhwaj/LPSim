from pathlib import Path
import json,hashlib,zipfile,shutil
root=Path(__file__).resolve().parents[3];source=Path(__file__).parent;art=root/'artifacts/shadow-study-20261104';out=root/'reports/shadow-study-20261104'
for name in ['cost-receipt.json','local-checks.json','causality-check.json']:
 if (art/name).exists():shutil.copy2(art/name,out/name)
files=[p for p in out.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='manifest.json'];manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files};manifest.update({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()});(out/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append(out/'manifest.json')
with zipfile.ZipFile(out/'report-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(out)))
with zipfile.ZipFile(out/'replay-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in [source,art/'inputs',root/'artifacts/multigpu-preparation-20261027/source']:
  for p in folder.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(root)))
 for p in files:z.write(p,str(p.relative_to(root)))
 for p in (art/'cloud').iterdir():
  if p.is_file():z.write(p,str(p.relative_to(root)),compress_type=zipfile.ZIP_STORED)
 for name in ['launch.json','launch-reservation.json','budget.json','local-checks.json','causality-check.json']:
  if (art/name).exists():z.write(art/name,str((art/name).relative_to(root)))
print('Report and replay bundles created')
