from pathlib import Path
import struct,json,random,math,shutil,hashlib
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'artifacts/prior-study-20261103/inputs';net=ROOT/'data/networks/berkeley_junction_study';out.mkdir(parents=True,exist_ok=True)
for f in ['model.txt','geometry.json','route_pool.json']:shutil.copy2(net/f,out/f)
g=json.loads((net/'geometry.json').read_text());pool=json.loads((net/'route_pool.json').read_text());roads=g['roads'];CAR=struct.Struct('<102i5f');cases=[]
weights=[1+.3*sum(max(0,roads[e]['lanes']-1)+max(0,roads[e]['speed']/11.176-1) for e in r['edges'])/len(r['edges']) for r in pool]
# Fixed center known before either seed: busiest route-pool road, not observed traffic.
visits=[0]*len(roads)
for r in pool:
 for e in r['edges']:visits[e]+=1
center=roads[max(range(len(roads)),key=lambda i:visits[i])]['start']
local=[1+8*math.exp(-sum((x-y)**2 for x,y in zip(roads[r['edges'][0]]['start'],center))/(2*450**2)) for r in pool]
for split,seed in [('calibration',3301),('test',3302)]:
 for n in [4096,16384]:
  for pattern in ['distributed','concentrated','mixed','bursty']:
   rng=random.Random(f'{seed}:{n}:{pattern}');name=f'{split}_{pattern}_{n}';d=out/name;d.mkdir(exist_ok=True)
   w=[a*(b if pattern=='concentrated' else (.5+.5*b if pattern=='mixed' else 1)) for a,b in zip(weights,local)]
   rows=[]
   for route in rng.choices(pool,weights=w,k=n):
    departure=rng.uniform(0,480)
    if pattern=='concentrated':departure=rng.uniform(0,180)
    if pattern=='mixed':departure=rng.uniform(0,180) if rng.random()<.5 else rng.uniform(0,480)
    if pattern=='bursty':departure=max(0,min(479.99,rng.choice([30,150,270,390])+rng.gauss(0,10)))
    rows.append((departure,route))
   rows.sort(key=lambda z:z[0]);raw=bytearray(struct.pack('<if',n,0))
   for departure,r in rows:
    arrays=sum((r[k]+[0]*(32-len(r[k])) for k in ['edges','lanes','moves']),[])
    raw.extend(CAR.pack(0,0,len(r['edges']),-1,0,0,*arrays,departure,0,0,0,0))
   (d/'snapshot.bin').write_bytes(raw);cases.append(dict(name=name,n=n,pattern=pattern,split=split,seed=seed))
(out/'cases.json').write_text(json.dumps(cases,indent=2));(out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file() and p.name!='sha256.json'},indent=2))
print('Prepared',len(cases),'frozen cases')
