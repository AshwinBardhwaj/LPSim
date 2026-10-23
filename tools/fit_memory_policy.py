"""Fit only on distributed/concentrated demand and non-held-out regions.
Forecast inputs contain only network geometry and requested routes/departures.
Evaluation configs are frozen before separate allocation timing repetitions.
"""
from pathlib import Path
import json,time,hashlib,argparse
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=r/'artifacts/current/memory-policy-20261022');args=p.parse_args();a=args.output;b=r/'artifacts/current/berkeley-block-imbalance-20261019';g=json.loads((r/'data/networks/berkeley_junction_study/geometry.json').read_text());roads=g['roads'];moves=g['movements'];regions=np.array(list(map(int,(r/'data/networks/berkeley_junction_study/regions.txt').read_text().split())));ids=sorted(set(regions.tolist()));nr=len(roads);nreg=int(regions.max())+1;lanes=np.bincount(regions,weights=[x['lanes'] for x in roads],minlength=nreg);base_bytes=nr*8*4+nreg*2*4
cases=json.loads((b/'prepared.json').read_text())['cases'];forecasts={};forecast_ms={};times=[60,180,300,450]
for c in cases:
 begin=time.perf_counter();out=np.zeros((4,nreg,2))
 with (b/'cases'/c['name']/'scenario.txt').open() as f:
  next(f);next(f)
  for line in f:
   tok=line.split();arrival=float(tok[0]);length=int(tok[1]);edges=list(map(int,tok[2:2+length]));mids=list(map(int,tok[2+2*length:2+3*length]))
   for k,edge in enumerate(edges):
    road=roads[edge];duration=road['length']/road['speed']+(8 if k+1<length else 0);end=arrival+duration;region=regions[edge]
    for wi,t in enumerate(times):
     if arrival<=t<end:out[wi,region,0]+=1
    arrival=end
    if k+1<length:
     m=moves[mids[k]];end=arrival+m['length']/(7 if m['turn']==0 else 4)
     for wi,t in enumerate(times):
      if arrival<=t<end:out[wi,region,1]+=1
     arrival=end
 forecasts[c['name']]=out;forecast_ms[c['name']]=(time.perf_counter()-begin)*1000
(a/'prior_inputs.json').write_text(json.dumps(dict(description='Isolated-route travel with fixed 8-second junction wait; no held-out observed states',forecast_ms=forecast_ms,counts={k:v.tolist() for k,v in forecasts.items()}),indent=2))
rows=[]
for f in (a/'indices').glob('*.csv'):
 if '.features.' in f.name:continue
 pattern,n,seed,t,batch=f.stem.split('_');d=pd.read_csv(f);d['pattern']=pattern;d['n']=int(n);d['case']=f'{pattern}_{n}_{seed}';d['t']=int(t);d['batch']=int(batch);rows.append(d)
d=pd.concat(rows,ignore_index=True);assert d.exact.all();w=d.groupby(['case','pattern','n','t','batch','variant']).median(numeric_only=True).reset_index();w.to_csv(a/'index_windows.csv',index=False)
obs={}
for f in (a/'indices').glob('*.features.csv'):
 stem=f.name.removesuffix('.features.csv');pattern,n,seed,t,batch=stem.split('_');x=pd.read_csv(f);obs[(f'{pattern}_{n}_{seed}',int(t))]=x.set_index('region')
training=[];costs={}
for (case,pattern,n,t),group in w.groupby(['case','pattern','n','t']):
 med=group.groupby('variant').median(numeric_only=True);baseline=med.loc['compact','wall_ms'];costs[(case,t)]={}
 for region in ids:
  cost=baseline-med.loc[f'region_{region}','wall_ms'];costs[(case,t)][region]=float(cost)
  if pattern=='bursty' or region%5==0:continue
  observed=obs[(case,t)].loc[region];prior=forecasts[case][times.index(t),region];training.append(dict(case=case,t=t,region=region,lanes=lanes[region],active=observed.road_active,connectors=observed.connectors,prior_active=prior[0],prior_connectors=prior[1],benefit_ms=cost))
train=pd.DataFrame(training);train.to_csv(a/'policy_training.csv',index=False)
def features(active,connector,lane):return np.array([active,connector,active*connector/max(1,lane),lane],dtype=float)
def fit(kind):
 X=np.stack([features(x.prior_active if kind=='prior' else x.active,x.prior_connectors if kind=='prior' else x.connectors,x.lanes) for x in train.itertuples()]);mean=X.mean(0);scale=X.std(0);scale[scale==0]=1;Z=np.column_stack([np.ones(len(X)),(X-mean)/scale]);pen=np.eye(5)*1.;pen[0,0]=0;coef=np.linalg.solve(Z.T@Z+pen,Z.T@train.benefit_ms.to_numpy());return dict(mean=mean.tolist(),scale=scale.tolist(),coef=coef.tolist())
models={k:fit(k) for k in ['prior','current_state']}
def predict(model,active,connector,lane):
 x=(features(active,connector,lane)-model['mean'])/model['scale'];return float(np.r_[1,x]@model['coef'])
model_meta=dict(models=models,training_cases=sorted(train.case.unique()),excluded_pattern='bursty',excluded_regions=[int(i) for i in ids if i%5==0],target='median complete-step milliseconds saved by a regional cache',ridge_penalty=1.,caveat='Regional savings are not guaranteed additive; selected unions are measured independently.')
(a/'policy_model.json').write_text(json.dumps(model_meta,indent=2))
assert not any(x.startswith('bursty') for x in model_meta['training_cases']);assert not any(train.region%5==0)
plans=[]
for c in cases:
 if c['pattern']!='bursty':continue
 case=c['name']
 for t in times:
  observed=obs[(case,t)];prior=forecasts[case][times.index(t)]
  for pool in [4096,8192,16384]:
   for scope in ['all','heldout']:
    candidates=[i for i in ids if scope=='all' or i%5==0];choices=[]
    for name in ['uniform','vehicle_count','current_state','prior','oracle']:
     begin=time.perf_counter();score={}
     for i in candidates:
      q=observed.loc[i];size=8*lanes[i]
      if name=='uniform':score[i]=1/(i+1) # Fixed spatial order, independent of test state.
      elif name=='vehicle_count':score[i]=q.road_active/size
      elif name=='current_state':score[i]=predict(models[name],q.road_active,q.connectors,lanes[i])/size
      elif name=='prior':score[i]=predict(models[name],prior[i,0],prior[i,1],lanes[i])/size
      else:score[i]=costs[(case,t)][i]/size
     chosen=[];used=0
     for i in sorted(candidates,key=lambda i:(-score[i],i)):
      size=int(8*lanes[i])
      if (name in ['uniform','vehicle_count'] or score[i]>0) and used+size<=pool:chosen.append(int(i));used+=size
     elapsed=(time.perf_counter()-begin)*1000;choices.append((name,chosen));plans.append(dict(case=case,t=t,pool=pool,scope=scope,policy=name,regions=chosen,used_bytes=(base_bytes+used if chosen else 0),budget=base_bytes+pool,selection_ms=elapsed,feature_ms=float(observed.feature_ms.iloc[0]) if name in ['vehicle_count','current_state'] else 0,forecast_ms=forecast_ms[case] if name=='prior' else 0,refresh_steps=2400 if name in ['prior','uniform'] else 120))
    (a/'configs'/f'{case}_{t}_{pool}_{scope}.txt').write_text(f'{base_bytes+pool} {len(choices)}\n'+''.join(f'{name} {len(v)} '+' '.join(map(str,v))+'\n' for name,v in choices))
(a/'allocation_plans.json').write_text(json.dumps(plans,indent=2));print('Training rows',len(train),'plans',len(plans),'model hash',hashlib.sha256((a/'policy_model.json').read_bytes()).hexdigest())
