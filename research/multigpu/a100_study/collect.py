from pathlib import Path
import tarfile,json,hashlib
root=Path('artifacts/a100-study-20261028')
for archive in root.glob('*/*.tar.gz'):
 target=archive.parent/'raw';target.mkdir(exist_ok=True)
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   if not (target/m.name).resolve().is_relative_to(target.resolve()) or m.issym() or m.islnk():raise ValueError(m.name)
  t.extractall(target)
 for manifest in target.glob('*/sha256.json'):
  for path,digest in json.loads(manifest.read_text()).items():
   if hashlib.sha256((manifest.parent/path).read_bytes()).hexdigest()!=digest:raise RuntimeError(path+' hash mismatch')
 print(archive,'verified')
