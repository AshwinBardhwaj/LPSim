from pathlib import Path
import zipfile,hashlib,json
R=Path('reports/local-scaling-20261028');A=Path('artifacts/local-scaling-20261028');S=Path('research/multigpu/local_scaling')
reports=[p for p in R.iterdir() if p.is_file() and p.suffix!='.zip' and p.name!='bundle-manifest.json']
files=set(reports)
files.update(p for p in S.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
files.update(p for p in A.rglob('*') if p.is_file() and 'raw' not in p.relative_to(A).parts)
files.update(p for p in Path('artifacts/multigpu-preparation-20261027/source').rglob('*') if p.is_file())
files.add(Path('research/multigpu/a100_study/concurrent.cu'));files.add(Path('research/multigpu/container-lock.json'));files.add(Path('research/multigpu/prepare.py'))
for net in ['berkeley_junction_study','sf_partial']:
 files.update(p for p in (Path('data/networks')/net).glob('*') if p.is_file() and p.suffix in ('.json','.csv','.txt'))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(R/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2));files.add(R/'bundle-manifest.json')
for name,items in [('presentation-bundle.zip',reports),('reproducibility-bundle.zip',files)]:
 with zipfile.ZipFile(R/name,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p in sorted(items):z.write(p,str(p),compress_type=zipfile.ZIP_STORED if p.name.endswith('.tar.gz') else zipfile.ZIP_DEFLATED)
 with zipfile.ZipFile(R/name) as z:assert z.testzip() is None
 print(name,(R/name).stat().st_size)
