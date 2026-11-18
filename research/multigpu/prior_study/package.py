from pathlib import Path
import hashlib,json,zipfile,shutil
ROOT=Path(__file__).resolve().parents[3];report=ROOT/'reports/prior-study-20261103';art=ROOT/'artifacts/prior-study-20261103';source=Path(__file__).parent
shutil.copy2(art/'cost-receipt.json',report/'cost-receipt.json')
shutil.copy2(ROOT/'reports/fellowship-ground-truth-20261103/verified-summary.md',report/'fellowship-ground-truth.md')
files=[p for p in report.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='manifest.json']
manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()})
(report/'manifest.json').write_text(json.dumps(manifest,indent=2));files.append(report/'manifest.json')
with zipfile.ZipFile(report/'report-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,str(p.relative_to(report)))
with zipfile.ZipFile(report/'replay-bundle.zip','w',zipfile.ZIP_DEFLATED) as z:
 for folder in [source,art/'inputs',ROOT/'artifacts/multigpu-preparation-20261027/source']:
  for p in folder.rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:z.write(p,str(p.relative_to(ROOT)))
 for p in files:z.write(p,str(p.relative_to(ROOT)))
 for p in (art/'cloud').iterdir():
  if p.is_file():z.write(p,str(p.relative_to(ROOT)),compress_type=zipfile.ZIP_STORED)
 for name in ['launch.json','launch-reservation.json','cost-receipt.json','source-sha256.json']:
  z.write(art/name,str((art/name).relative_to(ROOT)))
print('Bundles written')
