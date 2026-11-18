from pathlib import Path
import json,struct,time,sys
import numpy as np
from paper_policies import partition as paper
from prior_policies import partition as prior,build_features,describe
CAR=struct.Struct('<102i5f');POLICIES=['geography','paper_balanced','boundary_aware']
def prepare(inputs,out):
 g=json.loads((inputs/'geometry.json').read_text());roads=g['roads'];meta=[];features=[]
 for c in json.loads((inputs/'cases.json').read_text()):
  raw=(inputs/c['name']/'snapshot.bin').read_bytes();cars=[CAR.unpack_from(raw,8+i*CAR.size) for i in range(c['n'])];folder=out/c['name'];folder.mkdir(parents=True,exist_ok=True)
  forecast_start=time.perf_counter();arr=np.asarray(cars);route=arr[:,6:38].astype(int);valid=np.arange(32)[None,:]<arr[:,2,None]
  length=np.array([r['length'] for r in roads]);speed=np.array([r['speed'] for r in roads]);duration=np.where(valid,length[route]/speed[route]+10,0)
  leave=arr[:,102,None]+np.cumsum(duration,axis=1);enter=leave-duration;mid=np.arange(20)*30+15
  occupancy=np.array([np.bincount(route[((enter<=t)&(leave>t)&valid)],minlength=len(roads)) for t in mid])
  pending_roads=np.array([np.bincount(route[arr[:,102]>t,0],minlength=len(roads)) for t in mid])
  last=route[np.arange(len(cars)),arr[:,2].astype(int)-1]
  complete_roads=np.array([np.bincount(last[leave[:,-1]<=t],minlength=len(roads)) for t in mid])
  forecast_ms=1000*(time.perf_counter()-forecast_start)
  for policy in POLICIES:
   t=time.perf_counter()
   if policy=='boundary_aware':owners,m=prior(g,cars,policy,3)
   else:owners,m=paper(g,cars,policy,3)
   map_ms=1000*(time.perf_counter()-t);(folder/(policy+'.txt')).write_text('\n'.join(map(str,owners))+'\n')
   t=time.perf_counter();owners=np.asarray(owners);src=np.array([owners[r['a']] for r in roads]);dst=np.array([owners[r['b']] for r in roads]);cut=src!=dst
   predicted=np.array([np.bincount(dst,weights=x,minlength=3) for x in occupancy])
   ghost=np.array([np.bincount(src[cut],weights=x[cut],minlength=3) for x in occupancy])
   pending=np.array([np.bincount(dst,weights=x,minlength=3) for x in pending_roads])
   completed=np.array([np.bincount(dst,weights=x,minlength=3) for x in complete_roads])
   for window in range(20):
    for rank in range(3):
     rr=dst==rank;cross=cut&(src==rank)
     features.append(dict(case=c['name'],policy=policy,window=window,rank=rank,split=c['split'],pattern=c['pattern'],n=c['n'],
      expected_active=float(predicted[window,rank]),expected_ghosts=float(ghost[window,rank]),expected_pending=float(pending[window,rank]),expected_completed=float(completed[window,rank]),
      road_count=int(rr.sum()),junction_count=int((owners==rank).sum()),cut_roads=int(cross.sum()),cut_lanes=sum(r['lanes'] for i,r in enumerate(roads) if cross[i]),
      lane_meters=sum(r['length']*r['lanes'] for i,r in enumerate(roads) if rr[i]),time_s=float(mid[window])))
   feature_ms=1000*(time.perf_counter()-t);meta.append(dict(case=c['name'],policy=policy,map_ms=map_ms,feature_ms=feature_ms,forecast_ms=forecast_ms,**{'details':m}))
  print('maps',c['name'],flush=True)
 (out/'maps.json').write_text(json.dumps(meta,indent=2));(out/'features.json').write_text(json.dumps(features))
if __name__=='__main__':prepare(Path(sys.argv[1]),Path(sys.argv[2]))
