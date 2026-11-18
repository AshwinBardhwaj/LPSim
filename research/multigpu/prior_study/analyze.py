"""Held-out analysis; all model choices fixed in PROTOCOL.md before timing."""
from pathlib import Path
import json,sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[3];raw=Path(sys.argv[1]);out=ROOT/'reports/prior-study-20261103';out.mkdir(exist_ok=True)
runs=pd.DataFrame(json.loads((raw/'runs.json').read_text()));maps=pd.DataFrame(json.loads((raw/'maps/maps.json').read_text()));features=pd.DataFrame(json.loads((raw/'maps/features.json').read_text()));timings=[];windows=[];pops=[];regional=[];wire=[]
for r in runs.to_dict('records'):
 if r['steps']!=2400:continue
 d=raw/r['path'];m=json.loads((d/'metrics.json').read_text());base={k:r[k] for k in ['name','policy','repeat','split','pattern','n']};base['case']=base.pop('name')
 if not r['diagnostic']:
  timings.append(dict(**base,**m,wall_ms=r['wall_s']*1000));w=pd.read_csv(d/'steps.csv');w['window']=w.step//120
  for win,v in w.groupby('window'):
   windows.append(dict(**base,window=win,**{k:float(v[k].sum()) for k in w.columns if k.endswith('_ms')}))
 else:
  p=pd.read_csv(d/'populations.csv');p['window']=p.step//120
  for k,v in base.items():p[k]=v
  pops.append(p)
  t=pd.read_csv(d/'trace.csv');t['window']=t.step//120
  compute=t[(t.gpu_ms>=0)&(~t.component.isin(['peer_copies','pack_reservations']))]
  for (win,rank),v in compute.groupby(['window','rank']):regional.append(dict(**base,window=win,rank=rank,regional_gpu_ms=v.gpu_ms.sum()/4))
  w=pd.read_csv(d/'wire.csv');previous=0
  for v in w.to_dict('records'):
   wire.append(dict(**base,window=v['step']//120,wire_bytes=v['bytes']-previous));previous=v['bytes']
t=pd.DataFrame(timings);w=pd.DataFrame(windows);p=pd.concat(pops);reg=pd.DataFrame(regional);wire=pd.DataFrame(wire)
t=t.merge(maps[['case','policy','map_ms','feature_ms','forecast_ms']],on=['case','policy']);t['complete_ms']=t.wall_ms+t.map_ms
t.to_csv(out/'timings.csv',index=False);w.to_csv(out/'windows.csv',index=False);p.to_csv(out/'populations.csv',index=False);reg.to_csv(out/'regional-gpu.csv',index=False);wire.to_csv(out/'boundary-by-window.csv',index=False)
summary=t.groupby(['case','policy','split','pattern','n']).median(numeric_only=True).reset_index();summary.to_csv(out/'policy-summary.csv',index=False)
# Window targets use medians across independent uninstrumented timing repeats.
win=w.groupby(['case','policy','window']).median(numeric_only=True).reset_index()
fx=['expected_active','expected_ghosts','expected_pending','expected_completed','road_count','junction_count','cut_roads','cut_lanes','lane_meters','time_s','n']
def model_predict(df,target,cols,name):
 train=df.split=='calibration';X=df[cols].to_numpy(float);Y=df[target].to_numpy(float);mu=X[train].mean(axis=0);sd=X[train].std(axis=0);sd[sd<1e-9]=1
 Z=np.column_stack([np.ones(len(X)),(X-mu)/sd]);A=Z[train];penalty=np.eye(Z.shape[1])*10;penalty[0,0]=0
 start=time.perf_counter();coef=np.linalg.solve(A.T@A+penalty,A.T@np.log1p(Y[train]));fit_ms=(time.perf_counter()-start)*1000
 start=time.perf_counter();pred=np.maximum(0,np.expm1(Z@coef));inference_ms=(time.perf_counter()-start)*1000
 df=df.copy();df['prediction']=pred;df.to_csv(out/(name+'-predictions.csv'),index=False)
 y=Y[~train];yp=pred[~train];metrics=dict(mae=float(np.mean(abs(y-yp))),wape=float(np.sum(abs(y-yp))/max(np.sum(abs(y)),1e-12)),r2=float(1-np.sum((y-yp)**2)/max(np.sum((y-y.mean())**2),1e-12)),fit_ms=fit_ms,inference_ms=inference_ms)
 (out/(name+'-model.json')).write_text(json.dumps(dict(features=cols,mean=mu.tolist(),scale=sd.tolist(),coefficients=coef.tolist(),metrics=metrics),indent=2));return df,metrics
regional_df=features.merge(reg[['case','policy','window','rank','regional_gpu_ms']],on=['case','policy','window','rank'])
regional_pred,rm=model_predict(regional_df,'regional_gpu_ms',fx,'regional-compute')
ghost_df=features.merge(p[['case','policy','window','rank','ghosts']],on=['case','policy','window','rank'])
ghost_pred,gm=model_predict(ghost_df,'ghosts',fx,'regional-boundary-records')
# Compute persistence: prediction of next-window change in the SAME region.
regional_pred=regional_pred.sort_values(['case','policy','rank','window']);gr=regional_pred.groupby(['case','policy','rank']);regional_pred['actual_delta']=gr.regional_gpu_ms.diff();regional_pred['predicted_delta']=gr.prediction.diff();regional_pred.to_csv(out/'persistence.csv',index=False)
agg=[]
for key,v in features.groupby(['case','policy','window']):
 row=dict(zip(['case','policy','window'],key));row.update({k:v.iloc[0][k] for k in ['split','pattern','n']})
 for f in fx:
  row[f+'_sum']=v[f].sum();row[f+'_max']=v[f].max()
 agg.append(row)
a=pd.DataFrame(agg);af=[c for c in a.columns if c.endswith(('_sum','_max'))]
# Explicit policy intercepts represent known algorithm differences, not test observations.
for policy in ['paper_balanced','boundary_aware']:
 key='policy_'+policy;a[key]=(a.policy==policy).astype(float);af.append(key)
a=a.merge(win[['case','policy','window','total_ms']],on=['case','policy','window']).merge(wire[['case','policy','window','wire_bytes']],on=['case','policy','window'])
wp,wm=model_predict(a,'total_ms',af,'window-runtime');bp,bm=model_predict(a,'wire_bytes',af,'boundary-volume')
train=summary[summary.split=='calibration'];test=summary[summary.split=='test'];fixed=train.groupby('policy').complete_ms.sum().idxmin()
measured=train.loc[train.groupby(['pattern','n']).complete_ms.idxmin()].set_index(['pattern','n']).policy.to_dict()
startup=(t[t.split=='calibration'].assign(startup=lambda z:z.wall_ms-z.compute_ms).groupby('policy').startup.median()).to_dict()
predict=wp[wp.split=='test'].groupby(['case','policy']).prediction.sum().reset_index();predict['predicted_full_ms']=predict.prediction+predict.policy.map(startup);choices=predict.loc[predict.groupby('case').predicted_full_ms.idxmin()].set_index('case').policy.to_dict()
results=[];selection_start=time.perf_counter()
for case,v in test.groupby('case'):
 v=v.set_index('policy');r=v.iloc[0];mp=measured[(r['pattern'],r['n'])];pp=choices[case];hindsight=v.complete_ms.idxmin();cost=maps[maps.case==case];prior_overhead=cost.map_ms.sum()+cost.feature_ms.sum()+cost.forecast_ms.iloc[0]+(wm['inference_ms']+rm['inference_ms']+bm['inference_ms'])/8
 results.append(dict(case=case,pattern=r['pattern'],n=int(r['n']),fixed_policy=fixed,fixed_ms=v.loc[fixed,'complete_ms'],measured_policy=mp,measured_ms=v.loc[mp,'complete_ms'],prior_policy=pp,prior_ms=v.loc[pp,'wall_ms']+prior_overhead,prior_overhead_ms=prior_overhead,prior_supplied_ms=v.loc[pp,'complete_ms'],hindsight_policy=hindsight,hindsight_ms=v.loc[hindsight,'complete_ms']))
selection_ms=1000*(time.perf_counter()-selection_start);sel=pd.DataFrame(results);sel['measured_ms']+=selection_ms/8
rng=np.random.default_rng(3303);bounds={}
for label in ['measured','prior','prior_supplied','hindsight']:
 sel[label+'_gain_pct']=100*(1-sel[label+'_ms']/sel.fixed_ms);g=sel[label+'_gain_pct'].to_numpy();boot=np.mean(rng.choice(g,(10000,len(g)),replace=True),axis=1);lo,hi=np.quantile(boot,[.025,.975]);bounds[label]=dict(mean_gain_pct=float(g.mean()),ci95_pct=[float(lo),float(hi)],passes_gate=bool(g.mean()>=5 and lo>0))
sel.to_csv(out/'heldout-selection.csv',index=False)
# Allocation metrics use observed populations and local-stage event times.
allocation=[]
for (case,policy),v in p.groupby(['case','policy']):
 v=v.copy();v['active']=v.stopped+v.moving+v.connector
 by=v.groupby('window');ratios=by.active.max()/by.active.mean().replace(0,np.nan)
 g=reg[(reg.case==case)&(reg.policy==policy)].groupby('window').regional_gpu_ms
 q=summary[(summary.case==case)&(summary.policy==policy)].iloc[0]
 allocation.append(dict(case=case,policy=policy,active_imbalance_mean=float(ratios.mean()),active_imbalance_peak=float(ratios.max()),
  gpu_elapsed_imbalance_mean=float((g.max()/g.mean()).mean()),mean_ghost_records=float(by.ghosts.sum().mean()),peak_ghost_records=int(by.ghosts.sum().max()),
  wire_bytes=int(q.wire_bytes),ownership_transfers=int(q.migrations),complete_ms=float(q.complete_ms),map_ms=float(q.map_ms),setup_ms=float(q.setup_ms)))
pd.DataFrame(allocation).to_csv(out/'allocation-summary.csv',index=False)
observed=reg.merge(p[['case','policy','window','rank','pending','stopped','moving','connector','completed','ghosts','processed_records']],on=['case','policy','window','rank'])
observed['active']=observed.stopped+observed.moving+observed.connector
observed[observed.split=='test'][['regional_gpu_ms','active','pending','stopped','moving','connector','completed','ghosts','processed_records']].corr().to_csv(out/'descriptive-state-correlations.csv')


# Population variation, fastest policy stability (exploratory timings, not causal).
p['active']=p.stopped+p.moving+p.connector
pt=p.groupby(['case','policy','window']).sum(numeric_only=True).reset_index();popsummary=pt[pt.policy=='geography'].groupby('case').agg(peak_active=('active','max'),mean_active=('active','mean'),final_completed=('completed','last'),peak_pending=('pending','max'),peak_stopped=('stopped','max'));popsummary.to_csv(out/'traffic-summary.csv')
best=win.loc[win.groupby(['case','window']).total_ms.idxmin()].sort_values(['case','window']);streaks=[]
for case,v in best.groupby('case'):
 labels=v.policy.tolist();longest=cur=1
 for i in range(1,len(labels)):
  cur=cur+1 if labels[i]==labels[i-1] else 1;longest=max(longest,cur)
 streaks.append(dict(case=case,longest_winning_streak_s=longest*30,switches=sum(labels[i]!=labels[i-1] for i in range(1,len(labels)))))
best.to_csv(out/'window-winners.csv',index=False);pd.DataFrame(streaks).to_csv(out/'winner-persistence.csv',index=False)
calibration_ms=float(t[t.split=='calibration'].wall_ms.sum());persistence=regional_pred[(regional_pred.split=='test')&regional_pred.actual_delta.notna()]
findings=dict(fixed_policy=fixed,selection=bounds,models=dict(regional_compute=rm,window_runtime=wm,boundary_volume=bm,regional_boundary_records=gm),persistence_delta_mae_ms=float(np.mean(abs(persistence.actual_delta-persistence.predicted_delta))),persistence_constant_cost_mae_ms=float(np.mean(abs(persistence.actual_delta))),calibration_timing_total_ms=calibration_ms,calibration_cost_per_eight_test_runs_ms=calibration_ms/8,selector_cpu_ms=selection_ms,physical_checks=json.loads((raw/'checks.json').read_text()),notes=['Static ownership: no online repartition migration. Ordinary vehicle transfers are included.','Same-network unseen-seed evaluation only; not a causal or external-network validation.','CUDA event samples are not Nsight hardware counters; regional elapsed GPU stages include local index/list/memory work, excluding peer copies.','CPU inference timings measured on analysis host; map/features measured in A100 container.','Forecast assumes free-flow residence plus fixed 10s approach dwell; does not solve queues.'])
(out/'findings.json').write_text(json.dumps(findings,indent=2));print(json.dumps(findings,indent=2))
