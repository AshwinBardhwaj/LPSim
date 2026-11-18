from pathlib import Path
import sys,json,struct,random,math,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prepare import CAR,read_snapshot
from policies import partition
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'artifacts/local-scaling-20261028/inputs';OUT.mkdir(parents=True,exist_ok=True)
def prepare(name,network,n):
 folder=OUT/name;folder.mkdir(exist_ok=True);geo=json.loads((network/'geometry.json').read_text());routes=json.loads((network/'route_pool.json').read_text());roads=geo['roads'];by={}
 for route in routes:by.setdefault((route['edges'][0],route['lanes'][0]),[]).append(route)
 slots=[]
 for (rid,lane),rs in by.items():
  road=roads[rid]
  for pos in range(10,int(road['length'])-10,15):slots.append((rid,lane,pos))
 if len(slots)<n:raise ValueError(f'{name}: only {len(slots)} safe active slots for {n}')
 rng=random.Random(2810);rng.shuffle(slots)
 # Modest main-road bias without replacement. A fixed seed also nests populations.
 slots.sort(key=lambda s:rng.random()**(1/math.sqrt(max(1,roads[s[0]]['speed'])*roads[s[0]]['lanes'])),reverse=True)
 cars=[]
 for rid,lane,pos in slots[:n]:
  r=rng.choice(by[rid,lane]);car=[1,0,len(r['edges']),-1,0,0]+r['edges']+[0]*(32-len(r['edges']))+r['lanes']+[0]*(32-len(r['lanes']))+r['moves']+[-1]*(32-len(r['moves']))+[0.,float(pos),5.,0.,0.];cars.append(car)
 rng.shuffle(cars);raw=struct.pack('<if',n,180.)+b''.join(CAR.pack(*c) for c in cars);(folder/'snapshot.bin').write_bytes(raw)
 for k in [1,2,3]:
  for policy in ['geography','paper_balanced','paper_community']:
   owners,meta=partition(geo,cars,policy,k);(folder/f'{policy}_{k}.txt').write_text('\n'.join(map(str,owners))+'\n');(folder/f'{policy}_{k}.json').write_text(json.dumps(meta,indent=2))
 return dict(name=name,network=network.name,n=n,initial_active=n,kind='constructed valid road snapshot; then original timesteps, not calibrated demand',sha256=hashlib.sha256(raw).hexdigest())
if __name__=='__main__':
 cases=[]
 for name,net,counts in [('berkeley','berkeley_junction_study',[4096,8192]),('sf','sf_partial_local_scaling',[16384,32768,65536,98304])]:
  network=ROOT/'data/networks'/net
  for n in counts:
   case=prepare(f'{name}_{n}',network,n);cases.append(case);print(case,flush=True)
 (OUT/'cases.json').write_text(json.dumps(cases,indent=2))
