"""Complete-horizon timing and fidelity report; never treat approximation as exact."""
from pathlib import Path
import json,argparse,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=r/'artifacts/current/gating-multirate-20261022');p.add_argument('--output',type=Path,default=r/'reports/gating-multirate-20261022');args=p.parse_args();a=args.input;o=args.output;o.mkdir(parents=True,exist_ok=True)
valid=json.loads((a/'validation.json').read_text());timing=json.loads((a/'timing.json').read_text());rows=[]
for x in timing:
 if x['batch']<0:continue
 q=dict(case=x['case'],variant=x['variant'],n=int(x['case'].split('_')[1]),batch=x['batch'],process_s=x['elapsed_s'],exact_validation_final=x['exact_validation_final']);q.update(x['metrics']);q.update(x['experiment']);rows.append(q)
d=pd.DataFrame(rows);base=d.query("variant=='baseline'").set_index(['case','batch'])
d['speedup']=[base.loc[(x.case,x.batch)].calculation_seconds/x.calculation_seconds for x in d.itertuples()]
current_times=d.query("variant=='current'").set_index(['case','batch']).calculation_seconds
d['speedup_vs_current']=[current_times.loc[(x.case,x.batch)]/x.calculation_seconds for x in d.itertuples()]
d['process_speedup']=[base.loc[(x.case,x.batch)].process_s/x.process_s for x in d.itertuples()]
mask={f.stem:json.loads(f.read_text()) for f in (a/'masks').glob('*.json')}
d['oracle_preparation_ms']=[mask[x.case]['preparation_ms'] if x.variant=='oracle' else 0 for x in d.itertuples()]
d['charged_preparation_s']=d.calculation_seconds+d.oracle_preparation_ms/1000
d['charged_trace_s']=[x.charged_preparation_s+(mask[x.case]['trace_run_seconds'] if x.variant=='oracle' else 0) for x in d.itertuples()]
d.to_csv(o/'timings.csv',index=False)
s=d.groupby(['n','variant'])[['calculation_seconds','advance_gpu_ms','junction_gpu_ms','process_s','speedup','process_speedup','speedup_vs_current','charged_preparation_s','charged_trace_s']].median().reset_index();s.to_csv(o/'timing_summary.csv',index=False)
gates=d.groupby(['case','variant']).speedup.agg(['min','median','max']).reset_index();gates['passes_20pct']=gates['min']>=1.2;process_min=d.groupby(['case','variant']).process_speedup.min();gates['process_min']=[process_min.loc[(x.case,x.variant)] for x in gates.itertuples()];gates['process_passes_20pct']=gates.process_min>=1.2;gates.to_csv(o/'decision_gates.csv',index=False)
vr=[]
for x in valid:
 q=dict(case=x['case'],variant=x['variant'],exact_reference=x['exact_reference'],exit_code=x['exit_code']);q.update(x['metrics']);q.update(x['experiment']);vr.append(q)
v=pd.DataFrame(vr);v['reused_fraction']=v.reused_acceleration_updates/v.road_updates.replace(0,np.nan);v.to_csv(o/'validation.csv',index=False)
# Every archived 30-second state plus final state, all requested stable vehicle IDs.
def read(path):
 data=path.read_bytes();return np.frombuffer(data[8:] if path.name.startswith('snapshot_') else data,dtype=np.uint32).reshape(-1,107)
fidelity=[]
for case in v.case.unique():
 b=a/'validation'/case/'trace';paths=sorted(b.glob('snapshot_*.bin'))+[b/'final.bin']
 for variant in ['current','oracle','multi2','multi4']:
  pos=[];speed=[];mismatch=0;total=0;matched=0
  for path in paths:
   x=read(path);y=read(a/'validation'/case/variant/path.name);xf=x.view(np.float32);yf=y.view(np.float32);ids=np.arange(len(x));xe=x[ids,6+x[:,1]];ye=y[ids,6+y[:,1]];xl=x[ids,38+x[:,1]];yl=y[ids,38+y[:,1]]
   same=(x[:,0]==y[:,0])&(x[:,1]==y[:,1])&(xe==ye)&(xl==yl)&(x[:,3]==y[:,3]);mismatch+=int((~same).sum());total+=len(x);active=same&np.isin(x[:,0],[1,2]);matched+=int(active.sum());pos.extend(np.abs(xf[active,103]-yf[active,103]).tolist());speed.extend(np.abs(xf[active,104]-yf[active,104]).tolist())
  ref=v[(v.case==case)&(v.variant=='trace')].iloc[0];cur=v[(v.case==case)&(v.variant==variant)].iloc[0]
  fidelity.append(dict(case=case,variant=variant,vehicle_samples=total,matched_active_samples=matched,state_route_mismatch_fraction=mismatch/total,position_p95_m=float(np.percentile(pos,95)) if pos else 0,position_max_m=max(pos,default=0),speed_p95_mps=float(np.percentile(speed,95)) if speed else 0,completion_delta=int(cur.completed-ref.completed),entry_delta=int(cur.entries-ref.entries)))
f=pd.DataFrame(fidelity);f.to_csv(o/'fidelity.csv',index=False)
# Full-step population context; no substitution of requested demand for active work.
pops=[]
for case in v.case.unique():
 x=pd.read_csv(a/'validation'/case/'trace'/'step_metrics.csv');pops.append(dict(case=case,requested=int(x.requested.iloc[0]),median_road_active=float((x.road_moving+x.road_stopped).median()),peak_road_active=int((x.road_moving+x.road_stopped).max()),median_connectors=float(x.connector.median()),median_pending=float(x.pending.median()),actual_request_fraction=mask[case]['actual_requests']/mask[case]['total_requested_updates'],oracle_candidate_fraction=mask[case]['candidate_updates']/mask[case]['total_requested_updates']))
pop=pd.DataFrame(pops);pop.to_csv(o/'population_and_eligibility.csv',index=False)
sections=[('What was tested','Two distinct ideas: exact removal of inactive junction request work; approximate multi-rate acceleration for isolated road interiors. Both use the frozen compact-motion implementation, the same Berkeley routes/departures and fixed 72-second controller. No learned network prior is trained. The oracle is an offline-known candidate schedule, not a deployable prediction policy. A future predictor would need conservative coverage or a cheap fallback for unexpected requests; a false-negative candidate can change traffic behavior.'),('Exact gating mechanism','Baseline request generation evaluates N requested records and sorts N keys. Current-state compaction still evaluates N records, filters invalid keys, then sorts only actual requests. The oracle uses the union of request-eligible vehicle IDs over each two-second interval, collected from an instrumented reference trajectory. It evaluates and sorts only that candidate set, rechecking actual signal/conflict eligibility every step. Stable-ID priority, movement protections, spawn handling, connector motion and car-following remain unchanged. Mask transfers every eight steps are charged in loop time. This tests a specific junction-request gate, not all possible empty-topology or motion skipping.'),('Multi-rate mechanism and limitations','For eligible vehicles, acceleration decisions occur every 0.5 or 1 second instead of every 0.25 second. Positions and speeds still integrate every 0.25 second using held acceleration. Eligibility is checked each tick: no lane leader, no incoming-lane connector, and enough distance to the road end for the rate horizon plus margin. Newly interacting vehicles immediately resume full equations. Leader/context indexes, compact refresh, junction processing and safety bounds remain at the base rate. This conservatively targets acceleration work, not all vehicle updates; it is not a test of unrestricted regional time advancement. Eligibility excludes even distant road leaders, so negative results do not cover every free-flow approximation. No 40-60% work reduction is assumed. Earlier compact-state matched snapshots placed the entire motion kernel at roughly 7-13% of step time: a large gain would likely require also avoiding index/context or other stages. This experiment does not implement that broader architectural change. Approximate trajectories are compared against the original model, not asserted identical.'),('Measurement protocol','Nine full 600-second runs per variant: distributed, concentrated and bursty demand at 2,048, 8,192 and 16,384 requests. Main-road-biased demand remains unchanged. Instrumented validation and trace generation are separate from timings. One complete-horizon warmup and three shuffled independent process repetitions. Jetson clocks are fixed during runs and restored afterward. GPU stage totals and host simulation-loop time are measured; CPU diagnostics and trajectory output are excluded. Process elapsed includes loading/setup. Every timed final state must match its corresponding validated variant. Compilation: -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17.'),('Oracle accounting and success gate','The supplied oracle includes applying its schedule, copying masks and all remaining simulation work. Separate columns charge CPU mask preparation and the instrumented run used to discover the schedule. The latter is a practical construction cost, not an unavoidable future-predictor cost. A 20% speedup must hold in all three repetitions for a case to clear the ambitious structural gate. Engine-loop and full-process gates are reported separately; a loop win is not automatically an end-to-end process win. Compare oracle to current-state compaction before attributing opportunity to a prior. Failure of this candidate does not bound every structural gating design.'),('Fidelity and validation interpretation','Exact gating must match reference states, events, timeseries, saved snapshots and final states. Approximate variants undergo every-step gap, conflict, invalid-state and stopped-connector checks. Boundary gap diagnostics use lane extrema with the same zero-failure threshold. Fidelity tables use every saved 30-second snapshot plus final state: positional errors apply only when the same vehicle occupies the same road/connector and route position in both variants. State/route mismatch is reported separately to avoid hiding divergent trajectories. A lack of invariant violations is not real-traffic calibration or proof of safety on unseen scenarios.')]
sections.insert(0,('Decision', 'Exact request gating is worth further investigation at the larger populations. Current-state compaction yields about 1.13x engine-loop speedup at 8,192 and 16,384 requests, while the supplied offline schedule yields about 1.22-1.23x. The oracle retains only about an 8% incremental engine opportunity over compaction at those populations, before forecasting and coverage-check costs. Six of nine cases clear the 20% loop gate, but none clears the 20% full-process gate in every repeat. At 2,048 requests, compaction loses and oracle benefit is small. No network-prior benefit has been demonstrated. A conservative current-state candidate schedule is the next strong baseline before training a predictor.'))
sections.append(('Approximation error tail', 'Both guarded multi-rate variants are approximately performance-neutral. They reuse acceleration on only 0.9-13.0% of road updates across cases. Completion deltas are small (-1 to +2 vehicles), but this does not imply trajectory equivalence: maximum matched-container position error reaches about 241 metres, and up to about 0.5% of all requested-vehicle samples differ in state or route position. All implemented invariants passed, but this fidelity cost has no demonstrated runtime payoff. Do not promote these approximate variants. Broader regional multi-rate schemes that also avoid index maintenance remain untested.'))
# Numeric conclusions are generated from the actual completed experiments.
for variant in ['current','oracle','multi2','multi4']:
 q=d[d.variant==variant];g=gates[gates.variant==variant];z=v[v.variant==variant];err=f[f.variant==variant]
 text=f'Median paired full-loop speedup: {q.speedup.median():.3f}x. Loop cases clearing 20% in every repeat: {g.passes_20pct.sum()}/{len(g)}; full-process cases: {g.process_passes_20pct.sum()}/{len(g)}. Validation failures: {(z.exit_code!=0).sum()}/{len(z)}. Exact reference cases: {z.exact_reference.sum()}/{len(z)}. '
 if variant.startswith('multi'):text+=f'Median acceleration-reuse fraction of road updates: {z.reused_fraction.median():.1%}. Maximum state/route mismatch across a case: {err.state_route_mismatch_fraction.max():.2%}. Completion-count deltas range from {err.completion_delta.min()} to {err.completion_delta.max()}. '
 if variant=='oracle':text+=f'Median speedup versus current-state compaction: {q.speedup_vs_current.median():.3f}x. This is the remaining opportunity for a predictor to recover, before prediction overhead.'
 sections.append((variant+' measured outcome',text))
def markdown(df):
 def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])
md=['# Exact gating and guarded multi-rate results','']
for title,body in sections:md += ['## '+title,'',body,'']
md+=['## Full-horizon timing medians','',markdown(s),'','## Fidelity','',markdown(f),''];(o/'README.md').write_text('\n'.join(md))
with PdfPages(o/'gating-multirate.pdf') as pdf:
 for title,body in sections:
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.9,title,fontsize=19,weight='bold');fig.text(.07,.8,'\n'.join(textwrap.wrap(body,105)),va='top',fontsize=12,linespacing=1.7);fig.text(.07,.05,'Berkeley | 600 s | dt=0.25 s | exact gating versus approximate dynamics',fontsize=9,color='gray');pdf.savefig(fig);plt.close(fig)
 def table(title,df):
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,fontsize=16,pad=20);x=df.copy()
  for col in x.select_dtypes(include='number'):x[col]=x[col].map(lambda z:f'{z:.3f}' if isinstance(z,float) else str(z))
  tab=ax.table(cellText=x.values,colLabels=x.columns,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.6);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
 table('600-second simulations: seconds and paired speedup',s[['n','variant','calculation_seconds','process_s','speedup','process_speedup']])
 table('Actual workload and fraction of requested records needing requests',pop[['case','median_road_active','peak_road_active','actual_request_fraction','oracle_candidate_fraction']])
 for variant in ['multi2','multi4']:table(variant+': errors versus base-rate simulation',f[f.variant==variant][['case','state_route_mismatch_fraction','position_p95_m','position_max_m','completion_delta']])
 fig,ax=plt.subplots(figsize=(9,5))
 for variant,g in s.groupby('variant'):ax.plot(g.n.to_numpy(),g.speedup.to_numpy(),'o-',label=variant)
 ax.axhline(1,color='gray');ax.axhline(1.2,color='green',linestyle='--');ax.set(xlabel='Requested trips',ylabel='Paired full-loop speedup',title='Exact work gating and approximate acceleration updates');ax.legend();fig.tight_layout();fig.savefig(o/'speedup.png',dpi=180);pdf.savefig(fig);plt.close(fig)
print(s[['n','variant','calculation_seconds','speedup']].to_string(index=False))
