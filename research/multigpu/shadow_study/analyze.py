from pathlib import Path
import json,sys,csv
import numpy as np,pandas as pd
from propose import read,road_ids
ROOT=Path(__file__).resolve().parents[3];raw=Path(sys.argv[1]);out=ROOT/'reports/shadow-study-20261104';out.mkdir(exist_ok=True)
runs=json.loads((raw/'runs.json').read_text());proposals=pd.read_json(raw/'proposals.json');rows=[];diagnostics=[];population=[];halves=[]
for r in runs:
 if r['steps']!=240:continue
 d=raw/r['path'];m=json.loads((d/'metrics.json').read_text());row={**r,**m};steps=pd.read_csv(d/'steps.csv');row.update({'stage_'+k:float(steps[k].sum()) for k in steps.columns if k.endswith('_ms')});rows.append(row)
 if not r['diagnostic']:
  for half,v in steps.groupby(steps.step//120):halves.append(dict(case=r['case'],decision_s=r['decision_s'],policy=r['policy'],repeat=r['repeat'],half=int(half),total_ms=v.total_ms.sum()))
 if r['diagnostic']:
  p=pd.read_csv(d/'populations.csv')
  for col in ['case','policy','decision_s']:p[col]=r[col]
  population.append(p)
  trace=pd.read_csv(d/'trace.csv');trace=trace[(trace.gpu_ms>=0)&(trace.component!='peer_copies')]
  for rank,v in trace.groupby('rank'):diagnostics.append(dict(case=r['case'],policy=r['policy'],decision_s=r['decision_s'],rank=rank,mean_sampled_gpu_ms=v.gpu_ms.sum()/8))
t=pd.DataFrame(rows);timing=t[~t.diagnostic.astype(bool)];summary=timing.groupby(['case','decision_s','policy']).median(numeric_only=True).reset_index();result=[]
for (case,decision),v in summary.groupby(['case','decision_s']):
 v=v.set_index('policy');base=v.loc['fixed'];capture=pd.read_csv(raw/(case+'__capture')/'capture.csv');observations=capture[capture.time_s.isin([decision-30,decision])].observation_ms.sum()
 diag=t[(t.case==case)&(t.decision_s==decision)&(t.policy=='fixed')&t.diagnostic.astype(bool)].compute_ms.iloc[0]
 excess=max(0,float(diag-base.compute_ms))
 shared=json.loads((raw/'shared-preparation.json').read_text())['topology_load_ms']/3 # charge one topology load per independent run, three decisions
 for policy in ['reactive','forecast']:
  p=proposals[(proposals.case==case)&(proposals.time_s==decision)&(proposals.policy==policy)].iloc[0];candidate=v.loc[policy];overhead=p.construction_ms+observations+excess+shared;headroom=base.compute_ms-candidate.compute_ms-overhead
  result.append(dict(case=case,decision_s=int(decision),policy=policy,fixed_step_ms=base.compute_ms,replay_step_ms=candidate.compute_ms,raw_gain_pct=100*(1-candidate.compute_ms/base.compute_ms),construction_ms=p.construction_ms,shared_preparation_ms=shared,observation_ms=observations,diagnostic_excess_ms=excess,pre_migration_ms=candidate.compute_ms+overhead,migration_budget_ms=headroom,pre_migration_gain_pct=100*headroom/base.compute_ms,changed_vehicle_authorities=p.changed_vehicle_authorities,changed_junctions=p.changed_junctions,vehicle_payload_lower_bound_bytes=p.vehicle_payload_lower_bound_bytes,cold_process_ms=candidate.wall_ms+overhead,fixed_cold_process_ms=base.wall_ms,current_active=p.current_active,current_pending=p.current_pending,fixed_wire_bytes=base.wire_bytes,replay_wire_bytes=candidate.wire_bytes,wire_reduction_pct=100*(1-candidate.wire_bytes/max(base.wire_bytes,1)),ordinary_transfers=candidate.migrations))
r=pd.DataFrame(result);r.to_csv(out/'window-comparison.csv',index=False);timing.to_csv(out/'timings.csv',index=False);summary.to_csv(out/'timing-summary.csv',index=False);pd.concat(population).to_csv(out/'populations.csv',index=False);pd.DataFrame(diagnostics).to_csv(out/'per-gpu-cost.csv',index=False);proposals.to_csv(out/'proposals.csv',index=False)
# Forecast accuracy is evaluated only AFTER proposals/replays are complete.
quality=[];nr=len(json.loads((ROOT/'artifacts/shadow-study-20261104/inputs/geometry.json').read_text())['roads'])
for (case,t0),v in r.groupby(['case','decision_s']):
 measured=[]
 for stamp in [t0,t0+30,t0+60]:
  _,c=read(raw/(case+'__capture')/f'snapshot_{stamp}.bin');ids=road_ids(c);active=(c[:,0]==1)|(c[:,0]==2);measured.append(np.bincount(ids[active],minlength=nr))
 target=(measured[0]+2*measured[1]+measured[2])/4
 for policy in ['reactive','forecast']:
  prediction=np.load(raw/'maps'/case/str(t0)/(policy+'-forecast.npz'))['predicted_road_active'];quality.append(dict(case=case,decision_s=t0,policy=policy,absolute_error=float(np.abs(target-prediction).sum()),actual_residents=float(target.sum()),wape=float(np.abs(target-prediction).sum()/max(target.sum(),1))))
half=pd.DataFrame(halves).groupby(['case','decision_s','policy','half']).total_ms.median().reset_index();half.to_csv(out/'half-window-times.csv',index=False)
q=pd.DataFrame(quality);q.to_csv(out/'forecast-quality.csv',index=False)
rng=np.random.default_rng(3405)
def bounds(values):
 b=rng.choice(values,(10000,len(values)),replace=True).mean(axis=1);lo,hi=np.quantile(b,[.025,.975]);return dict(mean_pct=float(np.mean(values)),ci95_pct=[float(lo),float(hi)],passes_pre_migration_gate=bool(np.mean(values)>=5 and lo>0))
findings={}
for policy,v in r.groupby('policy'):
 findings[policy]=dict(raw=bounds(v.groupby('case').raw_gain_pct.mean().to_numpy()),pre_migration=bounds(v.groupby('case').pre_migration_gain_pct.mean().to_numpy()),positive_budget_windows=int((v.migration_budget_ms>0).sum()),max_migration_budget_ms=float(v.migration_budget_ms.max()),median_construction_ms=float(v.construction_ms.median()),forecast_wape=float(q[q.policy==policy].absolute_error.sum()/q[q.policy==policy].actual_residents.sum()))
pivot=r.pivot(index=['case','decision_s'],columns='policy',values='pre_migration_ms');relative=100*(1-pivot.forecast/pivot.reactive);findings['forecast_vs_reactive']=bounds(relative.groupby('case').mean().to_numpy());findings['checks']=json.loads((raw/'checks.json').read_text());findings['limitations']=['Previously inspected demand cases; exploratory mechanism pilot.','No actual live migration: budget is a break-even amount, not measured migration time.','Rank-average cost attribution is not direct per-junction profiling and can include fixed topology costs.','Observation charges include checkpoint/validation overhead; diagnostic excess is a noisy positive-difference estimate.','No Nsight counters collected; separate CUDA events only.','Forecast is causal trend plus scheduled departures, not a trained model; overdue pending admissions and downstream route propagation are not explicitly forecast.']
(out/'findings.json').write_text(json.dumps(findings,indent=2));print(json.dumps(findings,indent=2))
