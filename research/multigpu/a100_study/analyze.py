"""Aggregate repeated timing separately from diagnostic and profiler runs."""
from pathlib import Path
import json,csv,io
import pandas as pd
import numpy as np
root=Path('artifacts/a100-study-20261028');out=Path('reports/a100-study-20261028');out.mkdir(exist_ok=True)
rows=[];diag=[];pop=[];alloc=[];mem=[];states=[]
for manifest in root.glob('*/raw/*/runs.json'):
 base=manifest.parent
 for record in json.loads(manifest.read_text()):
  p=base/record['path']
  if not (p/'metrics.json').exists():continue
  metrics=json.loads((p/'metrics.json').read_text());row={**record,**metrics,'artifact':str(p)}
  row['n']=int(row['case'].split('_')[1]) if row['case'].startswith('reference') else int(row['case'].split('_')[-1]);row['pattern']=row['case'].split('_')[1]
  row['label']='single' if row['backend']=='single' else row['policy'];rows.append(row)
  if (p/'memory.csv').exists():
   df=pd.read_csv(p/'memory.csv');df['case']=row['case'];df['label']=row['label'];df['repeat']=row['repeat'];mem.extend(df.to_dict('records'))
  if row['instrument'] and row['repeat']==7:
   steps=pd.read_csv(p/'steps.csv');pops=pd.read_csv(p/'populations.csv');pops['case']=row['case'];pops['label']=row['label'];pop.extend(pops.to_dict('records'))
   for stamp in [180,360,540]:
    subset=steps[(steps.time>=stamp)&(steps.time<stamp+10)]
    if subset.empty:continue
    r={'case':row['case'],'n':row['n'],'pattern':row['pattern'],'label':row['label'],'window_start':stamp}
    for col in steps.columns:
     if col not in ['step','time']:r[col]=subset[col].mean()
    if row['backend']=='peer':
     motion=subset[[f'gpu{i}_motion_ms' for i in range(3)]];junction=subset[[f'gpu{i}_junction_ms' for i in range(3)]]
     r['balance_only_ideal_saved_ms']=(motion.max(axis=1)-motion.mean(axis=1)+junction.max(axis=1)-junction.mean(axis=1)).mean()
     r['balance_only_ideal_pct']=100*r['balance_only_ideal_saved_ms']/r['total_ms']
     pp=pops[(pops.time>=stamp)&(pops.time<stamp+10)].copy();pp['active']=pp.stopped+pp.moving+pp.connector
     totals=pp.groupby('step').active.agg(['max','mean']);r['active_max_mean']=np.mean(totals['max']/totals['mean'].replace(0,np.nan));r['mean_total_active']=pp.groupby('step').active.sum().mean();r['mean_ghosts']=pp.groupby('step').ghosts.sum().mean()
    diag.append(r)
    for rank in range(3 if row['backend']=='peer' else 1):
     pr=pops[(pops.time>=stamp)&(pops.time<stamp+10)&(pops['rank']==rank)]
     if pr.empty:continue
     state={'case':row['case'],'n':row['n'],'pattern':row['pattern'],'label':row['label'],'window_start':stamp,'rank':rank,'gpu_motion_ms':subset[f'gpu{rank}_motion_ms'].mean(),'gpu_junction_ms':subset[f'gpu{rank}_junction_ms'].mean()}
     for key in ['pending','stopped','moving','connector','completed','ghosts','allocated_slots','requests','admissions','denied_space','denied_conflict_or_yield']:
      if key in pr:state[key]=pr[key].mean()
     state['active']=state['stopped']+state['moving']+state['connector'];states.append(state)
 for p in base.glob('*__allocation/allocation.json'):
  for r in json.loads(p.read_text()):alloc.append({'case':p.parent.name.split('__')[0],**r})
allruns=pd.DataFrame(rows);allruns.to_csv(out/'all-runs.csv',index=False)
if not allruns.empty:
 timings=allruns[(allruns['verify']==0)&(allruns.instrument==0)&allruns.repeat.between(1,5)]
 summary=[]
 for (case,label),g in timings.groupby(['case','label']):
  x=g.compute_wall_ms.to_numpy();rng=np.random.default_rng(2710);boot=np.median(rng.choice(x,(5000,len(x))),axis=1)
  ar=next((a for a in alloc if a['case']==case and a['policy']==label),{})
  summary.append(dict(case=case,n=g.n.iloc[0],pattern=g.pattern.iloc[0],label=label,repeats=len(g),median_ms=np.median(x),min_ms=x.min(),max_ms=x.max(),ci_low_ms=np.quantile(boot,.025),ci_high_ms=np.quantile(boot,.975),setup_median_ms=g.setup_ms.median(),assignment_ms=ar.get('construction_ms',0),cold_median_ms=np.median(x+g.setup_ms.to_numpy())+ar.get('construction_ms',0),wire_bytes_median=g.wire_bytes.median(),control_bytes_median=g.control_bytes.median(),migrations_median=g.ownership_transfers.median()))
 pd.DataFrame(summary).to_csv(out/'timing-summary.csv',index=False)
 # Independent calibration/evaluation: repeat 0 selects; repeats 1–5 evaluate.
 S=pd.DataFrame(summary);oracles=[]
 for case,g in allruns[(allruns['verify']==0)&(allruns.instrument==0)&(allruns.repeat==0)&(allruns.backend=='peer')].groupby('case'):
  if g.label.nunique()!=3:continue
  chosen=g.sort_values(['compute_wall_ms','label']).iloc[0]
  evalrow=S[(S.case==case)&(S.label==chosen.label)]
  if evalrow.empty:continue
  calibration=g.compute_wall_ms.sum()+g.setup_ms.sum()+sum(a.get('construction_ms',0) for a in alloc if a['case']==case)
  oracles.append({'case':case,'selected_policy':chosen.label,'calibration_choice_ms':chosen.compute_wall_ms,'evaluation_median_ms':evalrow.median_ms.iloc[0],'evaluation_cold_ms':evalrow.cold_median_ms.iloc[0],'calibration_cost_ms':calibration,'charged_oracle_ms':calibration+evalrow.cold_median_ms.iloc[0]})
 pd.DataFrame(oracles).to_csv(out/'restricted-oracle.csv',index=False)
pd.DataFrame(states).to_csv(out/'state-cost.csv',index=False)
pd.DataFrame(diag).to_csv(out/'diagnostic-windows.csv',index=False);pd.DataFrame(pop).to_csv(out/'populations.csv',index=False);pd.DataFrame(alloc).to_csv(out/'allocations.csv',index=False);pd.DataFrame(mem).to_csv(out/'memory.csv',index=False)
profiles=[];profile_states=[]
for manifest in root.glob('*/raw/*/profiles.json'):
 for record in json.loads(manifest.read_text()):
  p=manifest.parent/record['path']/'metrics.csv'
  pp=p.parent/'pre-step-populations.json'
  if pp.exists():
   for state in json.loads(pp.read_text()):profile_states.append({k:record[k] for k in ['case','time','policy','backend','exit_code']}|state)
  if not p.exists():continue
  lines=p.read_text().splitlines();header=next((i for i,s in enumerate(lines) if s.startswith('"ID"')),None)
  if header is None:continue
  parsed=list(csv.reader(io.StringIO('\n'.join(lines[header:]))));columns=parsed[0]
  if 'Metric Name' in columns:
   for values in parsed[1:]:
    r=dict(zip(columns,values))
    if r.get('Metric Name'):profiles.append({**record,**r})
  else:
   # Nsight raw-page export is a wide table: one units row, one row/kernel.
   units=parsed[1]
   wanted=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__thread_inst_executed_per_inst_executed.ratio','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_wait_per_warp_active.pct','l1tex__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread','smsp__inst_executed.sum']
   for values in parsed[2:]:
    r=dict(zip(columns,values))
    if not r.get('ID','').isdigit():continue
    for metric in wanted:
     if metric in columns:
      i=columns.index(metric);profiles.append({**record,**{k:r.get(k,'') for k in ['ID','Process ID','Kernel Name','Device','Context','Stream','Grid Size','Block Size']},'Metric Name':metric,'Metric Unit':units[i],'Metric Value':r[metric]})
pd.DataFrame(profiles).to_csv(out/'nsight-raw.csv',index=False)
pd.DataFrame(profile_states).to_csv(out/'nsight-populations.csv',index=False)
print('runs',len(rows),'windows',len(diag),'Nsight metric rows',len(profiles))
