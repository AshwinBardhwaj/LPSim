"""Read-only audit of frozen snapshots, scan instrumentation and sampled SASS."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts/current/berkeley-state-study-20261014';O=ROOT/'reports/berkeley-scan-audit-20261014';O.mkdir(exist_ok=True)
dtype=np.dtype({'names':['state','k','n','mid','edges','lanes','moves','pos','v'],'formats':['i4','i4','i4','i4',('i4',32),('i4',32),('i4',32),'f4','f4'],'offsets':[0,4,8,12,24,152,280,412,416],'itemsize':428})
f=(A/'inputs/berkeley_junction_study/model.txt').open();nr,nm,nj=map(int,f.readline().split());roads=np.array([list(map(float,f.readline().split())) for _ in range(nr)]);moves=np.array([list(map(float,f.readline().split())) for _ in range(nm)])
format_,entries=f.readline().split();assert format_=='SPARSE'
conflicts=[set() for _ in range(nm)]
for _ in range(int(entries)):
 a,b=map(int,f.readline().split());conflicts[a].add(b)
rows=[];sass=[]
for p in sorted((A/'replays').glob('*/*/instrumented.json.counts.csv')):
 case=p.parent.parent.name;t=int(p.parent.name);snap=A/'runs'/case/'0'/f'snapshot_{t}.bin';raw=snap.read_bytes();cars=np.frombuffer(raw[8:],dtype=dtype);n=len(cars);ix=np.arange(n);edge=cars['edges'][ix,cars['k']];lane=cars['lanes'][ix,cars['k']];road=cars['state']==1;conn=np.flatnonzero(cars['state']==2)
 counts=list(csv.DictReader(p.open())); assert len(counts)==n
 rcount=np.array([int(x['receiving_candidates']) for x in counts]);ccount=np.array([int(x['connector_candidates']) for x in counts]);assert np.all(ccount[road]==n)
 relevant=np.zeros(n,dtype=int)
 for j in conn:
  m=moves[cars['mid'][j]];relevant+=(road&(edge==int(m[1]))&(lane==int(m[3])))
 query=np.flatnonzero(rcount>0);mid=cars['moves'][query,cars['k'][query]];unique=np.unique(mid)
 repeated_saved=sum(int(rcount[query[mid==m]].sum()-rcount[query[mid==m]][0]) for m in unique)
 for m in unique:
  assert len(set(rcount[query[mid==m]].tolist()))==1
  movement=moves[m]
  blocked=road&(edge==int(movement[2]))&(lane==int(movement[4]))&(cars['pos']<14.5)
  for j in conn:
   if cars['mid'][j]!=m and int(cars['mid'][j]) in conflicts[m]:blocked[j]=True
  hits=np.flatnonzero(blocked);expected=int(hits[0])+1 if len(hits) else n
  assert np.all(rcount[query[mid==m]]==expected),(case,t,int(m),expected)

 rows.append(dict(case=case,time=t,requested=n,road_active=int(road.sum()),connectors=len(conn),connector_candidates=int(ccount.sum()),relevant_connector_pairs=int(relevant.sum()),road_vehicles_with_relevant_connector=int((relevant>0).sum()),receiving_candidates=int(rcount.sum()),receiving_queries=len(query),unique_receiving_movements=len(unique),duplicate_receiving_candidates=repeated_saved,receiving_queries_over_30m=int((roads[edge[query],3]-cars['pos'][query]>30).sum()),snapshot_sha256=hashlib.sha256(raw).hexdigest()))
for p in sorted((A/'profiles').glob('*/*/source.csv')):
 records=list(csv.DictReader(p.read_text().splitlines()[2:]));line=None
 for i,r in enumerate(records):
  if r['Line No']:line=int(r['Line No'])
  if not r['Address'].startswith('0x'):continue
  samples=float(r['stall_long_sb'].replace(',','').replace('-','') or 0)
  if not samples:continue
  sass.append(dict(case=p.parent.parent.name,time=p.parent.name,line=line,address=r['Address'],instruction=r['Source'].strip(),long_scoreboard_samples=samples,preceding_instructions=' | '.join(x['Source'].strip() for x in records[max(0,i-6):i] if x['Address'].startswith('0x'))))
for name,data in [('scan_audit',rows),('instruction_audit',sass)]:
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
print(json.dumps([r for r in rows if r['time']==180 and r['case'] in ['distributed_2048_seed1401_mode0','concentrated_2048_seed1401_mode0']],indent=2))

(O/'validation.json').write_text(json.dumps({'snapshots':len(rows),'connector_scan_counts_match':True,'receiving_scan_counts_independently_reproduced':True,'scope':'Read-only audit; no kernel changes or new timing claims'},indent=2)+'\n')
