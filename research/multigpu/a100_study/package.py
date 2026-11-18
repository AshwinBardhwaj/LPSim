"""Package presentation and complete reproducibility evidence without duplicate extraction."""
from pathlib import Path
import hashlib,json,zipfile
root=Path.cwd();report=Path('reports/a100-study-20261028');art=Path('artifacts/a100-study-20261028');freeze=Path('artifacts/multigpu-preparation-20261027');source=Path('research/multigpu/a100_study')
reports=[p for p in report.iterdir() if p.is_file() and p.suffix not in ('.zip',) and p.name!='bundle-manifest.json']
files=set(reports)
files.update(p for p in source.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.update(p for p in art.rglob('*') if p.is_file() and 'raw' not in p.relative_to(art).parts and p.stat().st_size>0)
files.update(p for p in freeze.iterdir() if p.is_file())
for sub in ('source','inputs'):
 files.update(p for p in (freeze/sub).rglob('*') if p.is_file())
manifest={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(files)}
(report/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
files.add(report/'bundle-manifest.json')
for name,items in [('presentation-bundle.zip',reports),('reproducibility-bundle.zip',files)]:
 with zipfile.ZipFile(report/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
  for p in sorted(items):z.write(p,str(p),compress_type=zipfile.ZIP_STORED if p.name.endswith('.tar.gz') else zipfile.ZIP_DEFLATED)
 with zipfile.ZipFile(report/name) as z:
  assert z.testzip() is None
 print(name,(report/name).stat().st_size)
