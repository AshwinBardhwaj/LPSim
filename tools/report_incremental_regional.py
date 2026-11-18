"""Report exact independent index/regional experiments; no predictor claims."""
from pathlib import Path
import csv,json,hashlib,textwrap,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
r=Path(__file__).resolve().parents[1];a=r/'artifacts/current/incremental-regional-20261023';o=r/'reports/incremental-regional-20261023';o.mkdir(parents=True,exist_ok=True)
valid=json.loads((a/'index-validate.json').read_text());rows=[]
for x in json.loads((a/'index-timing.json').read_text()):
 if x['batch']<0:continue
 rows.append(dict(case=x['case'],policy=x['policy'],batch=x['batch'],n=int(x['case'].split('_')[1]),seconds=x['metrics']['calculation_seconds'],process_s=x['process_s'],exact=x['exact']))
d=pd.DataFrame(rows);base=d.query("policy=='bulk'").set_index(['case','batch']);d['speedup']=[base.loc[(x.case,x.batch),'seconds']/x.seconds for x in d.itertuples()];d['process_speedup']=[base.loc[(x.case,x.batch),'process_s']/x.process_s for x in d.itertuples()];d.to_csv(o/'index_timings.csv',index=False)
s=d.groupby(['n','policy'])[['seconds','process_s','speedup','process_speedup']].median().reset_index();s.to_csv(o/'index_summary.csv',index=False)
case=d.groupby(['case','policy']).agg(seconds=('seconds','median'),speedup=('speedup','median'),minimum_speedup=('speedup','min'),maximum_speedup=('speedup','max')).reset_index();case.to_csv(o/'index_by_case.csv',index=False)
diag=[]
for f in (a/'index-validation'/'-1').glob('*/incremental/index_metrics.csv'):
 q=pd.read_csv(f).iloc[1:];i=q.index_ms.sum();b=q.reference_bulk_ms.sum();candidate=np.minimum(q.index_ms,q.reference_bulk_ms).sum();diag.append(dict(case=f.parent.parent.name,incremental_ms=i,reference_ms=b,mean_changes=q.membership_changes.mean(),mean_dirty_lanes=q.dirty_lanes.mean(),optimistic_saved_index_fraction=1-candidate/min(i,b)))
pd.DataFrame(diag).to_csv(o/'index_diagnostics.csv',index=False)
pop=[]
for f in (a/'index-validation'/'-1').glob('*/incremental/step_metrics.csv'):
 q=pd.read_csv(f);pop.append(dict(case=f.parent.parent.name,mean_road_active=(q.road_stopped+q.road_moving).mean(),peak_road_active=int((q.road_stopped+q.road_moving).max()),mean_stopped=q.road_stopped.mean(),mean_connector=q.connector.mean(),mean_pending=q.pending.mean()))
pd.DataFrame(pop).to_csv(o/'populations.csv',index=False)
prep=pd.DataFrame(json.loads((a/'regional-prepared.json').read_text()));rr=[]
for family in ['regional','regional-deferred']:
 for x in prep.itertuples():
  for batch in range(3):
   f=a/family/f'{x.case}_{x.t}_{x.H}_{batch}.csv'
   q=pd.read_csv(f);q['family']=family;q['case']=x.case;q['t']=x.t;q['batch']=batch;q['n']=x.n;q['road_active']=x.road_active;q['certificate_ms']=x.certificate_ms;rr.append(q)
g=pd.concat(rr,ignore_index=True);med=g.groupby(['family','case','t','H','batch','variant'])[['gpu_ms','wall_ms','transfer_ms','cold_setup_ms','certificate_ms','local_ms','index_ms','n','selected','road_active']].median().reset_index();ref=med.query("variant=='bulk'").set_index(['family','case','t','H','batch'])
med['speedup']=[ref.loc[(x.family,x.case,x.t,x.H,x.batch),'wall_ms']/x.wall_ms for x in med.itertuples()]
med['charged_speedup']=[ref.loc[(x.family,x.case,x.t,x.H,x.batch),'wall_ms']/(x.wall_ms+(x.certificate_ms+x.transfer_ms if x.variant!='bulk' else 0)) for x in med.itertuples()]
med['cold_charged_speedup']=[ref.loc[(x.family,x.case,x.t,x.H,x.batch),'wall_ms']/(x.wall_ms+(x.certificate_ms+x.cold_setup_ms if x.variant!='bulk' else 0)) for x in med.itertuples()]
med['selected_fraction']=med.selected/med.road_active;g.to_csv(o/'regional_raw.csv',index=False);med.to_csv(o/'regional_windows.csv',index=False)
rs=med.groupby(['H','variant'])[['speedup','charged_speedup','selected_fraction','local_ms','wall_ms']].median().reset_index();rs.to_csv(o/'regional_summary.csv',index=False)
with PdfPages(o/'incremental-regional.pdf') as pdf:
 def page(title,paragraphs):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,title,fontsize=20);y=.86
  for para in paragraphs:
   lines=textwrap.fill(para,112);fig.text(.06,y,lines,fontsize=11,va='top');y-=.035*(lines.count('\n')+1)+.035
  pdf.savefig(fig);plt.close(fig)
 page('Exact incremental indexes and regional execution',[
 'Berkeley, 23 October 2026. Independent mechanisms against the frozen compact-motion/radix baseline. Distributed, concentrated and bursty demand; 2,048 / 8,192 / 16,384 requested vehicles; 600 seconds, dt=0.25 seconds, original 72-second controller. Requested population is not road-active population.',
 'Incremental index: retain lane lists and repair only lanes whose membership changed. All records are still scanned to detect membership; touched lanes are pruned before entrants are inserted. Full bulk rebuilding and a two-second periodic rebuild are separate controls. All construction and maintenance is charged.',
 'Validation: 27 full runs, every pre-motion leader ID checked against a fresh radix rebuild. Frozen trajectories, events, snapshots and invariant checks pass. Three separate timing repetitions per case/policy after warmup; final states are byte-identical. Timing-mode zero diagnostic counters are not validation evidence.',
 'Regional execution: whole roads conservatively certified independent for 2/4/8/20 timesteps. Original equations execute in shared memory; states are published each base timestep. A stronger deferred variant publishes only the window-final state and retains intermediate state locally; separate validation materializes all intermediate states. Selected vehicles are omitted from global indexes, motion and request work. Remaining global stages and resets stay in total time. Packed small-road and one-road-per-block variants are measured independently.',
 'This is a mechanism experiment, not evidence that a learned traffic prior helps. Compare any predictor against the best fixed/current-state rule. No new Nsight counter collection was performed.'])
 def table(title,frame,name):
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,pad=25,fontsize=17);t=ax.table(cellText=frame.astype(str).values,colLabels=frame.columns,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.65);fig.tight_layout();pdf.savefig(fig);fig.savefig(o/name,dpi=170);plt.close(fig)
 table('Full 600-second runs: median timing and speedup',s.round(3),'index_table.png')
 table('Regional windows: supplied mask versus charged certificate',rs.round(3),'regional_table.png')
 page('Interpretation and limits',[
 'Index diagnostics measure incremental construction first and reference rebuilding second. Their per-step minimum is only an optimistic, order-biased diagnostic candidate, not a timed switching policy. It excludes switching costs and cannot establish forecast value. Use the independent whole-run timings for performance conclusions.',
 'Regional certificate: excludes any road with a connector dependency, possible incoming request, pending departure, green admission or right-on-red admission during the horizon. Last-route vehicles cannot complete within the window. Selected positions are sufficiently far from road entries that receiving-space status stays unchanged. Whole-road groups contain at most 128 vehicles. This conservative candidate is not a maximal independence oracle.',
 'Certificate time is CPU Python analysis of current state and known routes/signals. Supplied-mask results omit that classification and list transfer; charged results include both. Cold allocation is recorded separately. A GPU certificate could be cheaper, but cannot repair a supplied-mask execution loss.',
 'Regional results are matched snapshot windows at 60, 180, 300 and 450 seconds, three process batches, ten timing repetitions per variant, randomized within each process. All regional variants are byte-compared at every intermediate timestep and every timed final state. They are not full-horizon regional timing runs.',
 'The two mechanisms are not combined. Regional comparisons use the original bulk baseline; if incremental indexing improves that baseline, a regional win against bulk alone is insufficient. No production default was changed.'])
 fig,axes=plt.subplots(1,2,figsize=(11.7,5));
 for policy in ['incremental','periodic8']:
  z=s[s.policy==policy];axes[0].plot(z.n.to_numpy(),z.speedup.to_numpy(),'o-',label=policy)
 axes[0].axhline(1,color='black',lw=.7);axes[0].set(xlabel='Requested vehicles',ylabel='Complete-loop speedup over bulk');axes[0].legend()
 for v in ['regional','packed','deferred']:
  z=rs[rs.variant==v];axes[1].plot(z.H.to_numpy()*.25,z.speedup.to_numpy(),'o-',label=v)
 axes[1].axhline(1,color='black',lw=.7);axes[1].set(xlabel='Local horizon (seconds)',ylabel='Window wall speedup, supplied mask');axes[1].legend();fig.tight_layout();pdf.savefig(fig);fig.savefig(o/'speedups.png',dpi=180);plt.close(fig)
summary=dict(index=s.to_dict('records'),regional=rs.to_dict('records'),validation_runs=len(valid),index_timing_runs=len(d),regional_timed_windows=len(g),regional_all_exact=bool(g.exact.all()))
(o/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
