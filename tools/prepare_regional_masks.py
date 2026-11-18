"""Conservative dependency certificate from current state, routes and known signals."""
from pathlib import Path
import json,struct,time
import numpy as np
r=Path(__file__).resolve().parents[1];a=r/'artifacts/current/incremental-regional-20261023';b=r/'artifacts/current/berkeley-block-imbalance-20261019';tokens=(r/'data/networks/berkeley_junction_study/model.txt').read_text().split();it=iter(tokens);nr,nm,nj=[int(next(it)) for _ in range(3)];roads=np.array([[float(next(it)) for _ in range(5)] for _ in range(nr)]);moves=np.array([[float(next(it)) for _ in range(9)] for _ in range(nm)]);cases=json.loads((b/'prepared.json').read_text())['cases'];(a/'region-masks').mkdir(exist_ok=True);rows=[]
def green(mid,t):
 m=moves[mid];p=(t+(36 if m[6]==1 else 0))%72
 return 24<=p<32 if m[5]==1 else p<20
for case in cases:
 for t in [60,180,300,450]:
  path=b/'runs'/case['name']/f'snapshot_{t}.bin';raw=path.read_bytes();n,time0=struct.unpack_from('if',raw);u=np.frombuffer(raw[8:],dtype=np.int32).reshape(n,107);f=u.view(np.float32);state=u[:,0];k=u[:,1];count=u[:,2];edge=u[np.arange(n),6+k];lane=u[np.arange(n),38+k];mid=u[:,3]
  for H in [2,4,8,20]:
   start=time.perf_counter();seconds=H*.25;forbidden=set();groups={};vmax=max(float(roads[:,4].max()),float(f[:,104].max()),7.);distance=vmax*seconds+.5*.5570409*seconds**2+6.4
   for i in range(n):
    st=int(state[i]);kk=int(k[i]);nn=int(count[i]);route=u[i,6:6+nn];route_moves=u[i,70:70+nn-1]
    if st==3:continue
    if st==0:
     if f[i,102]>t+seconds:continue
     forbidden.add(int(route[0]));kk=0;d=float(roads[route[0],3])
    elif st==2:
     m=moves[mid[i]];forbidden.add(int(m[1]));forbidden.add(int(m[2]));kk+=1;d=float(m[7]-f[i,103]+roads[route[kk],3])
    else:
     groups.setdefault(int(edge[i]),[]).append(i);d=float(roads[edge[i],3]-f[i,103])
    # A request can affect receiving-space before a connector is traversed.
    while kk+1<nn and d<=distance:
     forbidden.add(int(route[kk+1]));d+=float(moves[route_moves[kk],7]+roads[route[kk+1],3]);kk+=1
   selected=[]
   for road,ids in groups.items():
    if road in forbidden or len(ids)>128 or any(f[i,103]<14.5 for i in ids):continue
    safe=True
    for i in ids:
     if k[i]+1==count[i]:safe &= roads[road,3]-f[i,103]>distance
     else:
      m=int(u[i,70+k[i]]);safe &= not(moves[m,5]==-1 and moves[m,8]) and not any(green(m,t+h*.25) for h in range(H))
     if not safe:break
    if safe:
     ids=sorted(ids,key=lambda i:(int(lane[i]),float(f[i,103]),i));safe=all(lane[x]!=lane[y] or f[y,103]>f[x,103] for x,y in zip(ids,ids[1:]))
     if safe:selected.append(ids)
   selected.sort(key=lambda ids:0 if len(ids)<=8 else 1 if len(ids)<=32 else 2)
   filename=f"{case['name']}_{t}_{H}.txt";flat=[i for ids in selected for i in ids]
   (a/'region-masks'/filename).write_text(f'{H} {len(selected)} {len(flat)}\n'+''.join(str(len(ids))+' '+' '.join(map(str,ids))+'\n' for ids in selected))
   rows.append(dict(case=case['name'],t=t,H=H,n=n,road_active=int((state==1).sum()),selected=len(flat),roads=len(selected),certificate_ms=(time.perf_counter()-start)*1000,mask=filename,snapshot=str(path),distance_bound_m=distance));print(case['name'],t,H,len(flat),flush=True)
(a/'regional-prepared.json').write_text(json.dumps(rows,indent=2))
