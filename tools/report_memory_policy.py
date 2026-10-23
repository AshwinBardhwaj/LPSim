"""Summarize paired memory experiments; keep validation and timing separate."""
from pathlib import Path
import json, argparse, hashlib, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
r=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=r/'artifacts/current/memory-policy-20261022');p.add_argument('--output',type=Path,default=r/'reports/memory-policy-20261022');args=p.parse_args();a=args.input;o=args.output;o.mkdir(parents=True,exist_ok=True)
summary={};tables={};counts={}
for stage in ['storage','shared_final','indices','allocation']:
 rows=[]
 for f in sorted((a/stage).glob('*.csv')):
  if '.features.' in f.name:continue
  d=pd.read_csv(f);bits=f.stem.split('_');d['case']='_'.join(bits[:3]);d['n']=int(bits[1]);d['t']=int(bits[3]);d['batch']=int(bits[4]);d['sample']=f.stem
  if stage=='allocation':d['pool']=int(bits[5]);d['scope']=bits[6]
  rows.append(d)
 if not rows:raise RuntimeError('Missing '+stage)
 d=pd.concat(rows,ignore_index=True);assert d.exact.all();counts[stage]=len(d)
 keys=['sample','case','n','t','batch','variant']+(['pool','scope'] if stage=='allocation' else [])
 w=d.groupby(keys).median(numeric_only=True).reset_index()
 base=w[w.variant=='compact'].set_index('sample').wall_ms
 w['paired_speedup']=w['sample'].map(base)/w.wall_ms
 w.to_csv(o/f'{stage}_windows.csv',index=False)
 tables[stage]=w
 summary[stage]=w.groupby(['n','variant']).median(numeric_only=True).reset_index()
 summary[stage].to_csv(o/f'{stage}_summary.csv',index=False)
plans=pd.DataFrame(json.loads((a/'allocation_plans.json').read_text())).rename(columns={'policy':'variant'})
v=tables['allocation'].merge(plans,on=['case','t','pool','scope','variant'])
v['charged_ms']=v.wall_ms+(v.setup_ms+v.selection_ms+v.feature_ms_y)/v.refresh_steps+v.forecast_ms/2400
# Oracle measurement lower bound: actual timed characterization work, excluding validation/warmup.
measurement={}
for f in (a/'indices').glob('*.csv'):
 if '.features.' in f.name:continue
 bits=f.stem.split('_');key=('_'.join(bits[:3]),int(bits[3]));measurement[key]=measurement.get(key,0)+pd.read_csv(f).wall_ms.sum()
v['oracle_measurement_lower_bound_ms']=[measurement[(x.case,x.t)] if x.variant=='oracle' else 0 for x in v.itertuples()]
v['charged_with_measurement_ms']=v.charged_ms+v.oracle_measurement_lower_bound_ms/v.refresh_steps
base=tables['allocation'].query("variant == 'compact'").set_index('sample').wall_ms
v['charged_speedup']=v['sample'].map(base)/v.charged_ms
v['selected_regions']=v.regions.map(len)
v.to_csv(o/'allocation_charged_windows.csv',index=False)
allocation=v.groupby(['scope','pool','variant'])[['wall_ms','charged_ms','charged_with_measurement_ms','paired_speedup','charged_speedup','selected_regions']].median().reset_index();allocation.to_csv(o/'allocation_budgets.csv',index=False)
chains=[json.loads(f.read_text()) for f in (a/('storage_validation' if (a/'storage_validation').exists() else 'storage')).glob('*.chain.json')];assert len(chains)==36 and all(x['exact'] for x in chains)
compact=pd.read_csv(r/'reports/compact-motion-20261022/summary.csv')
sections=[]
def section(title,body):sections.append((title,body))
section('Decision', 'Freeze compact motion state as the improved baseline. Retained lane slots, the additional connector indexes and shared-memory staging all failed the predeclared 5% repeatable improvement gate; keep them experimental. No model-driven allocation benefit is established when the model chooses the compact baseline itself. These conclusions apply to the measured Berkeley workloads and implementations, not every possible regional cache.')
section('Experiment and correctness',f'Berkeley: 2,048 / 8,192 / 16,384 requested trips; distributed, concentrated and bursty main-road-biased demand. Matched snapshots at 60, 180, 300 and 450 seconds. Three independent process batches, three warmups and ten randomized timed repetitions per variant. Each timing includes index construction, context construction, refresh, motion, junction handling and final synchronization; trajectory diagnostics and output are excluded. All {sum(counts.values()):,} final-stage timed replays passed exact vehicle and reservation checks. Additionally 36 evolving 120-step retained-storage chains (4,320 steps) passed. Compact state separately passed 2,160 replays and nine full 600-second trajectories with 180 identical saved snapshots. Earlier milestones remain unchanged. One timestep is 0.25 seconds; CUDA uses -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17. Jetson clocks are fixed during each experiment and restored afterward. Counts are requested populations; road-active counts are recorded separately in the retained per-window feature files.')
section('Compact-state baseline', 'The original motion path copied the whole 428-byte vehicle record, including immutable route arrays. Compact motion reads 14 fields in a 56-byte-per-vehicle structure of arrays and writes only mutable fields back. Canonical routes remain allocated. Per-step refresh and unchanged stages are charged. This is an implementation improvement independent of any traffic prior. Commit: 89cb7c2f023e206c4719e83db34dd85fd9e01a0b. Enable LPSIM_COMPACT_MOTION=1 with the frozen configuration; it is opt-in.')
section('Persistent storage interpretation','Fresh lane slots rebuild the physical mapping every step. Retained slots reuse a lane ordering constructed from the snapshot 30 seconds earlier; current scalar values are refreshed into those slots every step. Stable IDs and canonical junction buffers preserve priority and reservation semantics. This is persistent compact slot mapping, not a fully resident lane-oriented rewrite of every simulator stage. Reported steady timings include refresh; retained setup is separately recorded and can be charged over 120 steps. The additional inverse/order storage and canonical refresh can outweigh locality. Full-record migration was already rejected in the earlier lane-layout milestone.')
section('Shared-memory controls','The ordered variant groups work in lane order without staging. Shared_lane stages 128 vehicle records as an array of structures; shared_soa stages the same 14 fields separately (7,168 bytes per block). Loads can reuse records inside their block tile; out-of-tile lookups use compact global state. Inverse-map construction, refresh and synchronization are included. Comparison against ordered isolates shared staging from grouping. This tests motion-state reuse, not a full neighborhood cache for every junction operation. No hardware-counter inference is made from timings alone.')
section('Selective lookup structures','Mandatory leader, connector-list and movement-availability indexes remain enabled for correctness. The optional index caches the nearest connector vehicle for each selected incoming lane, preserving position and stable-ID tie breaking. It is rebuilt from the pre-motion state each step. Characterization measures 15 individual regions, the full cache and four regional unions. This is the remaining connector lookup opportunity; it does not replace already-indexed receiving-space/conflict decisions. Fixed mapping and feature metadata cost 160,544 bytes; the candidate pool costs 8 bytes per selected lane. All 5,879 valid lanes require another 47,032 bytes.')
section('Prior and held-out evaluation','Ridge cost models are fitted only on distributed/concentrated runs and regions other than 0, 5, 10 and 15. Bursty demand and those regions are held out. The prior uses topology and supplied routes/departures, predicting isolated-route travel plus an uncalibrated eight-second junction wait; no observed test traffic enters that prior. The current-state model uses observed regional counts. Both estimate complete-step savings per byte. Whole-region greedy choices are evaluated with fresh timing samples. This is retrospective holdout on archived states, not prospective new-network validation. Candidate costs are not necessarily additive. A fixed uniform spatial order and a vehicle-count-per-byte rule provide baselines. Both fitted models selected no indexes for every tested plan; their tiny supplied-timing differences from compact are noise, not allocator speedups.')
section('Budget and overhead accounting','Extra slot pools are 4,096, 8,192 and 16,384 bytes, in addition to fixed mapping/feature metadata. The benchmark reserves the full pool; active required bytes are logged separately. Steady times include every cache rebuild. Charged times additionally amortize measured host map/transfer setup, selection and feature gathering over 120 steps (current/count/oracle) or 2,400 steps (prior/uniform); prior forecasting is amortized over 2,400 steps. These are explicit amortization assumptions, not a demonstrated online reallocation run. The oracle is a greedy measured-cost candidate, not an optimum. Its measurement-charged column adds a lower bound from all timed characterization samples for that window; warmups, validation and profiling are not included. Supplied oracle timings are intentionally optimistic.')
# Require each process batch to clear the threshold for a window-level win.
gates=[]
for stage in ['storage','shared_final','indices']:
 for name,g in tables[stage].groupby('variant'):
  if name=='compact':continue
  robust=g.groupby(['case','t']).paired_speedup.min()
  gates.append(dict(stage=stage,variant=name,windows=len(robust),repeatable_5pct_wins=int((robust>=1.05).sum()),median_paired_speedup=float(g.paired_speedup.median())))
pd.DataFrame(gates).to_csv(o/'decision_gates.csv',index=False)
section('Measured decision gate', 'Complete-step speedups must exceed 1.05 in every process batch of a matched window. Counts of qualifying windows, including individual caches and unions, are retained in decision_gates.csv. A local candidate win is not automatically a prior-policy win: the held-out selection must recover it after costs. Shared-memory staging is evaluated as a fixed block-local mechanism; because it has not established a broadly useful net benefit, the fixed-global-memory policy experiment selects optional lane indexes, not shared-memory capacity.')
section('Policy outcome', 'Median supplied-schedule speedups were approximately 1.000x for both fitted policies, which selected no caches; 0.991x for uniform, 0.990x for vehicle count and 0.999x for the oracle candidate. Charging the measured prototype overhead yields approximately 0.984x for current state and 0.758x for the prior. The prior forecast is an unoptimized CPU implementation; this cost is not a lower bound on forecasting. More fundamentally, no optional index configuration had a repeatable 5% complete-step win to recover. The charged oracle also pays characterization costs, separately from its optimistic supplied schedule.')
section('Limits and next decision','A useful prior needs an optional mechanism that wins in some conditions and loses in others. Rejecting uniformly costly caches does not establish that prediction is useful. Continue with the compact baseline and larger measured road-active populations only if the next mechanism controls substantial remaining cost. Do not infer bandwidth saturation, cache misses or scoreboard stalls from elapsed-time results. This study makes no new Nsight counter claim. Exact reference equivalence establishes preservation of the implemented model, not calibration against real traffic. Single-network, single-demand-seed coverage limits generalization.')
def markdown(df):
 def fmt(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])
md=['# Berkeley memory organization and allocation experiments','']
for title,body in sections:md += ['## '+title,'',body,'']
for stage in ['storage','shared_final']:
 q=summary[stage][['n','variant','wall_ms','motion_ms','paired_speedup']]
 md += ['## '+stage+' results','',markdown(q),'']
md+=['## Held-out budget policies','',markdown(allocation),'']
(o/'README.md').write_text('\n'.join(md))
with PdfPages(o/'memory-policy.pdf') as pdf:
 for title,body in sections:
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,title,fontsize=19,weight='bold');fig.text(.07,.82,'\n'.join(textwrap.wrap(body,105)),va='top',fontsize=12,linespacing=1.7);fig.text(.07,.05,'Berkeley memory-policy study | 2026-10-22 | measured times, exact model equivalence',fontsize=9,color='gray');pdf.savefig(fig);plt.close(fig)
 def page(title,frame):
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,pad=22,fontsize=16)
  frame=frame.copy()
  for col in frame.select_dtypes(include='number'):frame[col]=frame[col].map(lambda x:f'{x:.4f}' if isinstance(x,float) else str(x))
  tab=ax.table(cellText=frame.values,colLabels=frame.columns,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.6);pdf.savefig(fig,bbox_inches='tight');plt.close(fig)
 page('Frozen compact baseline: median complete-step milliseconds',compact[['n','variant','extract_ms','motion_ms','wall_ms']])
 for stage in ['storage','shared_final']:page(stage+': paired replay results',summary[stage][['n','variant','wall_ms','motion_ms','paired_speedup']])
 for scope in ['all','heldout']:page('Allocation: '+scope+' regions; charged milliseconds',allocation[allocation.scope==scope][['pool','variant','paired_speedup','charged_speedup','charged_with_measurement_ms']])
 fig,axes=plt.subplots(1,2,figsize=(11.7,5))
 for ax,stage in zip(axes,['storage','shared_final']):
  q=summary[stage]
  for name,g in q.groupby('variant'):ax.plot(g.n.to_numpy(),g.paired_speedup.to_numpy(),'o-',label=name)
  ax.axhline(1,color='gray');ax.axhline(1.05,color='green',linestyle='--');ax.set(xlabel='Requested vehicles',ylabel='Paired complete-step speedup',title=stage);ax.legend(fontsize=8)
 fig.tight_layout();fig.savefig(o/'memory-comparison.png',dpi=180);pdf.savefig(fig);plt.close(fig)
(o/'validation.json').write_text(json.dumps(dict(timed_exact_replays=counts,retained_chains=len(chains),retained_evolving_steps=4320,compact_trajectory_runs=9,compact_snapshots=180),indent=2))
print(json.dumps(counts));print(summary['shared_final'][['n','variant','wall_ms','paired_speedup']].to_string(index=False))
