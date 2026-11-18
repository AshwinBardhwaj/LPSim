"""Independently validate all distinct retained physical outputs, no GPU required."""
from pathlib import Path
import numpy as np,json,hashlib,struct
A=Path('artifacts/local-scaling-20261028');models={};seen={};results=[]
def model(name):
 if name not in models:
  with (A/'inputs/models'/name/'model.txt').open() as f:
   nr,nm,nj=map(int,f.readline().split());roads=np.array([list(map(float,f.readline().split())) for _ in range(nr)]);moves=np.array([list(map(float,f.readline().split())) for _ in range(nm)]);kind,nn=f.readline().split();assert kind=='SPARSE';conf=np.array([list(map(int,f.readline().split())) for _ in range(int(nn))])
  models[name]=roads,moves,conf
 return models[name]
def validate(path,network):
 raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
 if digest in seen:return
 n,t=struct.unpack_from('<if',raw);assert len(raw)==8+428*n
 ints=np.frombuffer(raw[8:],dtype='<i4').reshape(n,107);floats=ints.view('<f4');state=ints[:,0];k=ints[:,1];mid=ints[:,3];pos=floats[:,103];speed=floats[:,104]
 assert np.isfinite(pos).all() and np.isfinite(speed).all() and (pos>=0).all() and (speed>=0).all()
 assert ((state>=0)&(state<=3)).all() and ((k>=0)&(k<ints[:,2])).all()
 roads,moves,conf=model(network);road=ints[np.arange(n),6+k];lane=ints[np.arange(n),38+k];active=state==1;inside=state==2
 assert (pos[active]<=roads[road[active],3]+.001).all()
 assert ((mid[inside]>=0)&(mid[inside]<len(moves))).all()
 assert (speed[inside]>=.5).all() and (pos[inside]<=moves[mid[inside],7]+.001).all()
 def bounds(keys,positions,count):
  order=np.lexsort((positions,keys));ks=keys[order];xs=positions[order]
  assert not np.any((ks[1:]==ks[:-1])&(np.diff(xs)<6.499)),'same-lane gap'
  lo=np.full(count,np.inf);hi=np.full(count,-np.inf);np.minimum.at(lo,keys,positions);np.maximum.at(hi,keys,positions);return lo,hi
 near,tail=bounds(road[active]*8+lane[active],pos[active],len(roads)*8)
 entry,exit=bounds(mid[inside],pos[inside],len(moves));occupied=np.isfinite(entry)
 assert not np.any(occupied[conf[:,0]]&occupied[conf[:,1]]&(conf[:,0]!=conf[:,1])),'movement conflict'
 m=moves[occupied];incoming=m[:,1].astype(int)*8+m[:,3].astype(int);outgoing=m[:,2].astype(int)*8+m[:,4].astype(int)
 assert (roads[m[:,1].astype(int),3]-tail[incoming]+entry[occupied]>=6.499).all(),'incoming connector gap'
 assert (m[:,7]-exit[occupied]+near[outgoing]>=6.499).all(),'receiving connector gap'
 seen[digest]=True;results.append(dict(file=str(path),network=network,time=t,n=n,road=int(active.sum()),connector=int(inside.sum()),completed=int((state==3).sum()),sha256=digest,passed=True))
for runfile in A.glob('cloud*/raw/*/runs.json'):
 for run in json.loads(runfile.read_text()):
  path=runfile.parent/run['path']/'final.bin'
  if path.exists():validate(path,run['network'])
assert results
(A/'final-physical-validation.json').write_text(json.dumps(results,indent=2));print('PASS distinct final physical states',len(results))
