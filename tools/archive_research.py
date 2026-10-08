#!/usr/bin/env python3
"""One-time verified preservation and content-addressed cleanup of research outputs."""
import argparse,hashlib,json,os,shutil,stat,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ARCH=ROOT.parent/'research-archive'
DATE='20261008'
DEFS=[
 ('berkeley-baseline','Berkeley baseline and lane review','historical-provenance-incomplete',['artifacts/berkeley-lane-review'],[p.relative_to(ROOT).as_posix() for p in sorted((ROOT/'reports').glob('berkeley-*'))]),
 ('san-pablo-reduction','Reduced San Pablo network and synchronized signals','known-dense-occupancy-failures',['artifacts/san_pablo_university_20261004'],[]),
 ('occupancy-fix','Occupancy reservation and rollback fix','validated-recorded-cases',['artifacts/occupancy_fix_20261004'],['reports/san-pablo-occupancy-fix-20261006']),
 ('physical-junctions','Physical junctions and conservative reservations','validated-with-capacity-limitations',['artifacts/junction_behavior_20261006'],['reports/san-pablo-junctions-20261006']),
 ('lane-preserving-following','Lane-preserving connectors and safe following','validated-recorded-cases',['artifacts/junction_platoons_20261007'],['reports/san-pablo-platoons-20261007']),
 ('parallel-junctions','Parallel junction pipeline and serial equivalence','validated-recorded-cases',['artifacts/junction_parallel_20261007'],[]),
]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n')
def paths():
 for base,dirs,files in os.walk(ROOT,followlinks=False):
  dirs[:]=sorted(d for d in dirs if d not in ('.git','.agents','.codex','__pycache__','.pytest_cache'))
  for name in sorted(files):
   p=Path(base)/name
   if name in ('server.log','server.pid'):continue
   yield p

def backup():
 folder=ARCH/'backups'/DATE;folder.mkdir(parents=True,exist_ok=True);target=folder/'workspace.tar.gz'
 if (folder/'verified.json').exists():print('Backup already verified');return
 assert not target.exists(),'Partial backup exists; inspect before retrying'
 records=[];total=0;start=time.time()
 with tarfile.open(target,'w:gz',compresslevel=1,dereference=False) as tar:
  for p in paths():
   rel=p.relative_to(ROOT).as_posix();s=p.lstat()
   if p.is_symlink():records.append(dict(path=rel,link=os.readlink(p)));tar.add(p,arcname=rel,recursive=False);continue
   if not p.is_file():continue
   digest=sha(p);tar.add(p,arcname=rel,recursive=False);assert p.stat().st_size==s.st_size
   records.append(dict(path=rel,bytes=s.st_size,sha256=digest,mode=stat.S_IMODE(s.st_mode)));total+=s.st_size
   if len(records)%250==0:print('Backed up',len(records),'files;',round(total/1024**3,2),'GiB',flush=True)
 dump(folder/'inventory.json',records)
 expected={r['path']:r for r in records};seen=set()
 with tarfile.open(target,'r:gz') as tar:
  for member in tar:
   r=expected[member.name];seen.add(member.name)
   if 'link' in r:assert member.issym() and member.linkname==r['link'];continue
   h=hashlib.sha256();f=tar.extractfile(member)
   for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
   assert h.hexdigest()==r['sha256'],member.name
 assert seen==set(expected)
 dump(folder/'verified.json',dict(files=len(records),logical_bytes=total,archive_sha256=sha(target),elapsed_seconds=time.time()-start,same_device_backup=True))
 print('Backup verified:',len(records),'files',flush=True)

def apply():
 assert not (ARCH/'integrity.json').exists(), 'Archive is sealed; do not reapply migration'
 backupdir=ARCH/'backups'/DATE;assert (backupdir/'verified.json').exists(),'Verified backup required'
 expected={r['path']:r for r in json.loads((backupdir/'inventory.json').read_text())};moves=[]
 # Group reports/artifacts into milestone packages, preserving their original path names.
 for ident,title,status,artifacts,reports in DEFS:
  dest=ARCH/'milestones'/ident;dest.mkdir(parents=True,exist_ok=True)
  for rel in artifacts+reports:
   p=ROOT/rel;q=dest/rel
   if p.is_symlink():assert p.resolve()==q.resolve();continue
   if not p.exists():continue
   q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.move(str(p),str(q));p.symlink_to(q,target_is_directory=True);moves.append(dict(original=rel,archived=str(q)))
  # The complete input snapshot is referenced separately to avoid conflating old run provenance with today's checkout.
  dump(dest/'milestone.json',dict(id=ident,title=title,status=status,preservation_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),original_locations=artifacts+reports,backup='../../backups/'+DATE+'/workspace.tar.gz',note='Preservation commit is not necessarily the original run commit. Original provenance is retained inside each package.'))
 # Preserve loose generated files; source files and versioned networks stay in Git.
 ignored=subprocess.check_output(['git','ls-files','--others','--ignored','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
 for rel in ignored:
  if not rel:continue
  p=ROOT/rel
  if p.is_symlink() or not p.is_file():continue
  parts=Path(rel).parts
  if parts[0] in ('artifacts','reports','build','cache','.agents','.codex','lpsim_env','UNKNOWN.egg-info') or '__pycache__' in parts or '.pytest_cache' in parts:continue
  if rel not in expected:continue
  assert sha(p)==expected[rel].get('sha256'),rel
  dest=ARCH/'milestones/berkeley-baseline/legacy-exports'/rel;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();shutil.move(str(p),str(dest));moves.append(dict(original=rel,archived=str(dest)))
  if parts[0] in ('viz','docs'):p.symlink_to(dest)
 # Store an immutable copy of current inputs, including cache; original input files remain versioned.
 inputs=ARCH/('shared-inputs/checkout-'+subprocess.check_output(['git','rev-parse','--short','HEAD'],cwd=ROOT,text=True).strip())
 if not inputs.exists():
  shutil.copytree(ROOT/'data',inputs/'data');shutil.copytree(ROOT/'cache',inputs/'cache')
 moves=[dict(original=str(p.relative_to(ROOT)),archived=str(p.resolve())) for root in ('artifacts','reports','viz','docs') for p in (ROOT/root).rglob('*') if p.is_symlink() and str(p.resolve()).startswith(str(ARCH))]+moves
 dump(ARCH/'migration.json',moves)
 # Exact-content deduplication. File mode is part of the identity to preserve executables.
 objects=ARCH/'objects';records=[];logical=0;unique=0;count=0
 for base in (ARCH/'milestones',ARCH/'shared-inputs'):
  for p in sorted(base.rglob('*')):
   if not p.is_file() or p.is_symlink():continue
   if p.name in ('server.log','server.pid'):continue
   digest=sha(p);size=p.stat().st_size;executable=bool(p.stat().st_mode&0o111);key=digest+('.x' if executable else '.r');obj=objects/key[:2]/key
   obj.parent.mkdir(parents=True,exist_ok=True)
   if not obj.exists():shutil.copyfile(p,obj);assert sha(obj)==digest;obj.chmod(0o555 if executable else 0o444);unique+=size
   tmp=p.with_name(p.name+'.archive-link');os.link(obj,tmp);os.replace(tmp,p)
   records.append(dict(path=p.relative_to(ARCH).as_posix(),sha256=digest,bytes=size,object=obj.relative_to(ARCH).as_posix()));logical+=size;count+=1
   if count%500==0:print('Deduplicated',count,'files',flush=True)
 dump(ARCH/'inventory.json',records);dump(ARCH/'storage.json',dict(files=count,logical_bytes=logical,unique_bytes=unique,deduplicated_bytes=logical-unique))
 print('Moved and deduplicated',count,'files; saved',round((logical-unique)/1024**3,2),'GiB',flush=True)

def verify():
 rows=json.loads((ARCH/'inventory.json').read_text());cache={}
 for i,r in enumerate(rows):
  p=ARCH/r['path'];s=p.stat();key=(s.st_dev,s.st_ino)
  if key not in cache:cache[key]=sha(p)
  assert cache[key]==r['sha256'] and s.st_size==r['bytes'],r['path']
 dump(ARCH/'integrity.json',dict(passed=True,files=len(rows),unique_inodes=len(cache),verified_unix=time.time()))
 print('Archive integrity passed:',len(rows),'files')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['backup','apply','verify']);a=p.parse_args();globals()[a.action]()
