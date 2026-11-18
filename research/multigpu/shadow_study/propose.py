"""Causal shadow proposals: never reads checkpoints later than decision time."""
from pathlib import Path
import json,struct,time,itertools,csv
from dataclasses import replace
import numpy as np
from prior_policies import Features,partition
CAR=struct.Struct('<102i5f')
def read(path):
 raw=Path(path).read_bytes();n,t=struct.unpack_from('<if',raw)
 if len(raw)!=8+n*CAR.size:raise ValueError('Snapshot ABI')
 records=np.frombuffer(raw,offset=8,dtype=np.dtype([('i','<i4',(102,)),('f','<f4',(5,))]),count=n)
 return t,np.concatenate([records['i'],records['f']],axis=1)
def road_ids(c):return c[np.arange(len(c)),6+c[:,1].astype(int)].astype(int)
def make(inputs,capture,case,t,output,policy,geometry=None):
 start=time.perf_counter();g=geometry if geometry is not None else json.loads((inputs/'geometry.json').read_text());roads=g['roads'];nr=len(roads);dst=np.array([r['b'] for r in roads]);src=np.array([r['a'] for r in roads]);nj=int(max(dst.max(),src.max()))+1
 stamp,c=read(capture/f'snapshot_{t}.bin');past,p=read(capture/f'snapshot_{t-30}.bin');assert stamp==t and past==t-30 and len(c)==len(p)
 old=np.loadtxt(inputs/case/'fixed.txt',dtype=int);rid=road_ids(c);prid=road_ids(p);state=c[:,0].astype(int);previous=p[:,0].astype(int);active=(state==1)|(state==2);pactive=(previous==1)|(previous==2)
 count=lambda idx:np.bincount(idx,minlength=nr).astype(float)
 a=count(rid[active]);before=count(prid[pactive]);road=count(rid[state==1]);born=(previous==0)&(state!=0);born_origin=count(p[born,6].astype(int));departure=count(prid[pactive&((~active)|(prid!=rid))])
 own_j=dst[rid];connector=state==2
 for i in np.flatnonzero(connector):own_j[i]=g['movements'][int(c[i,3])]['junction']
 authority=old[own_j];visible=np.bincount(authority,minlength=3).astype(float);ghost_rank=old[src[rid[state==1]]];ghostmask=ghost_rank!=authority[state==1];visible+=np.bincount(ghost_rank[ghostmask],minlength=3)
 rank_ms=np.zeros(3)
 with (capture/'trace.csv').open() as f:
  for r in csv.DictReader(f):
   if (t-30)*4<=int(r['step'])<t*4 and float(r['gpu_ms'])>=0 and r['component']!='peer_copies':rank_ms[int(r['rank'])]+=float(r['gpu_ms'])/4
 if np.any(rank_ms<=0):raise ValueError('Missing preceding-window rank timings')
 unit=rank_ms/np.maximum(visible,1)
 passive=np.bincount(own_j[~active],minlength=nj).astype(float)
 current_work=np.bincount(own_j,minlength=nj).astype(float)
 predicted=a.copy();ghost_estimate=road.copy();scheduled=np.zeros(nr)
 if policy=='forecast':
  future=(state==0)&(c[:,102]>=t)&(c[:,102]<t+60);scheduled=count(c[future,6].astype(int))
  nonspawn_rate=(a-before-born_origin)/30
  capacity=np.array([max(1,r['lanes']*int(r['length']/6.5)) for r in roads],dtype=float)
  capacity=np.maximum(capacity,a) # preserve already-valid connector occupancy
  end=np.clip(a+60*nonspawn_rate+scheduled,0,capacity);total=end.sum()
  if total>len(c):end*=len(c)/total
  predicted=(a+end)/2;ghost_estimate=predicted*np.divide(road,a,out=np.ones(nr),where=a>0)
  passive=np.maximum(0,passive-.5*np.bincount(dst,weights=scheduled,minlength=nj))
  current_work=passive+np.bincount(dst,weights=predicted,minlength=nj)
  current_work*=len(c)/max(current_work.sum(),1)
 elif policy!='reactive':raise ValueError(policy)
 # Rank-average elapsed cost attribution is a proxy, not per-junction measurement.
 weighted=(current_work+1)*unit[old]
 features=Features(source=src,target=dst,lanes=np.array([r['lanes'] for r in roads],float),visits=np.zeros(nr),road_population=ghost_estimate,transition_per_step=departure/30*.25,owned_population=weighted,node_count=nj,expected_active=float(weighted.sum()),dt=.25)
 owners,meta=partition(g,[],'boundary_aware',3,features=features);owners=np.array(owners)
 # Label matching avoids counting a pure rank permutation as a required migration.
 def score(perm):
  aligned=np.asarray(perm)[owners];return int(np.count_nonzero(aligned[own_j]!=authority)),int(np.count_nonzero(aligned!=old)),perm
 perm=min(itertools.permutations(range(3)),key=score);owners=np.array(perm)[owners];changed=int(np.count_nonzero(owners[own_j]!=authority));changed_nodes=int(np.count_nonzero(owners!=old))
 meta.pop('statistics',None);meta.pop('expected_road_population',None);meta['forecast']='causal rolling traffic proxy' if policy=='forecast' else 'current traffic and preceding rank elapsed cost';meta['estimated_work_ms_by_rank']=np.bincount(owners,weights=weighted,minlength=3).tolist();meta['estimated_cut_road_records']=float(ghost_estimate[owners[src]!=owners[dst]].sum())
 output.mkdir(parents=True,exist_ok=True);target=output/(policy+'.txt');target.write_text('\n'.join(map(str,owners))+'\n')
 elapsed=1000*(time.perf_counter()-start)
 record=dict(case=case,time_s=t,policy=policy,construction_ms=elapsed,changed_vehicle_authorities=changed,changed_junctions=changed_nodes,vehicle_payload_lower_bound_bytes=432*changed,
  current_active=int(active.sum()),current_pending=int((state==0).sum()),current_completed=int((state==3).sum()),forecast_average_active=float(predicted.sum()),known_future_departures=int(scheduled.sum()),rank_elapsed_ms=rank_ms.tolist(),rank_visible=visible.tolist(),rank_ms_per_record=unit.tolist(),map_sha256=__import__('hashlib').sha256(target.read_bytes()).hexdigest(),partition_details=meta)
 (output/(policy+'.json')).write_text(json.dumps(record,indent=2));np.savez_compressed(output/(policy+'-forecast.npz'),current_road_active=a,predicted_road_active=predicted,current_road_ghost_candidates=road)
 return record
if __name__=='__main__':
 import sys
 print(make(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4]),Path(sys.argv[5]),sys.argv[6]))
