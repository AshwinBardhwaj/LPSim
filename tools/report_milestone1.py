#!/usr/bin/env python3
"""Create baseline tables, action-linked profiling plots and a PDF from raw evidence."""
import csv,json,re,statistics,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from milestone1_baseline import ART,ROOT,METRICS
OUT=ROOT/'reports/milestone1-baseline-20261008'
LABELS=['Kernel duration (µs)','L1/TEX throughput (% peak)','L2 throughput (% peak)','Uniform branch targets (%)','Fixed-latency wait (% warp cycles)','Eligible, not selected (% warp cycles)','SM throughput (% peak)','Achieved occupancy (%)','Eligible warps / scheduler cycle','Active threads / instruction (of 32)','Issue-active cycles (%)','Long-scoreboard stall (% warp cycles)']
DISPLAY=[0,6,7,8,9,10,11,4,5,3,1,2]
MODES=['controller72','always_permitted','corridor'];SHORT=['72 s','Permitted','Corridor']
def read(p):return list(csv.DictReader(p.open()))
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
 manifest=json.loads((ART/'manifest.json').read_text());gate=json.loads((ART/'gate.json').read_text());assert gate['passed'];summaries=[];profiles=[];actionrows=[];speed_samples={}
 for case in manifest['cases']:
  name=case['name'];base=ART/'runs'/name;records=[json.loads((base/f'parallel_{i}/validated.json').read_text())['metrics'] for i in range(3)];serial=[json.loads((base/f'serial_{i}/validated.json').read_text())['metrics'] for i in range(3)]
  series=read(base/'parallel_0/step_metrics.csv');states=read(base/'parallel_0/states.csv');speeds=np.array([float(r['speed']) for r in states]);road=np.array([float(r['speed']) for r in states if r['state']=='1']);m=records[0];speed_samples[name]=speeds
  summary=dict(case= name,mode=case['mode'],n=case['n'],completed=m['completed'],pending=m['pending'],active=m['active'],entries=m['entries'],completed_per_min=m['completed']/10,entries_per_min=m['entries']/10,road_stopped_sample_fraction=float(np.mean(road<.5)),speed_quantiles_mps=np.quantile(speeds,[0,.1,.5,.9,1]).tolist(),time_mean_road_stopped=float(np.mean([float(r['road_stopped']) for r in series])),time_mean_road_moving=float(np.mean([float(r['road_moving']) for r in series])),time_mean_pending=float(np.mean([float(r['pending']) for r in series])),gpu_ms_median=statistics.median(r['advance_gpu_ms']+r['junction_gpu_ms'] for r in records),gpu_ms_range=[min(r['advance_gpu_ms']+r['junction_gpu_ms'] for r in records),max(r['advance_gpu_ms']+r['junction_gpu_ms'] for r in records)],serial_gpu_ms_median=statistics.median(r['advance_gpu_ms']+r['junction_gpu_ms'] for r in serial),wall_s_median=statistics.median(r['wall_seconds'] for r in records),advance_ms_median=statistics.median(r['advance_gpu_ms'] for r in records),junction_ms_median=statistics.median(r['junction_gpu_ms'] for r in records),checks_passed=True)
  summaries.append(summary)
  for t in manifest['profile_times']:
   folder=ART/'profiles'/name/str(t);data=read(folder/'kernels.csv');units=data.pop(0);assert len(data)==6
   state=json.loads((folder/'sample.json').read_text())['pre_step_population_and_step_actions']
   context=[r for r in series if t-5<=float(r['time'])<t+5]
   actionrows.append(dict(case=name,**state,context_10s_entries_per_min=sum(int(r['entries']) for r in context)*6,context_10s_completions_per_min=sum(int(r['completed_step']) for r in context)*6))
   for row in data:
    kernel=re.search(r'::(\w+)\(',row['Kernel Name']).group(1);values={k:float(row[k].replace(',','')) for k in METRICS};values[METRICS[0]]*= {'ns':.001,'us':1,'µs':1,'ms':1000,'s':1e6}[units[METRICS[0]]]
    profiles.append(dict(case=name,mode=case['mode'],n=case['n'],time=t,kernel=kernel,block=row['Block Size'],grid=row['Grid Size'],**values,**{k:float(v) for k,v in state.items() if k not in ('time','advance_gpu_ms','junction_gpu_ms')}))
 # Evidence remains in ART; report exports are convenient derived tables.
 for file,rows in [('summary.csv',summaries),('profile_metrics.csv',profiles),('profile_actions.csv',actionrows)]:
  with (OUT/file).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 dump(OUT/'summary.json',summaries);dump(OUT/'manifest.json',manifest)
 pages=[]
 with PdfPages(OUT/'milestone1-baseline.pdf') as pdf:
  def save(fig,name,caption=''):
   fig.text(.045,.025,caption,fontsize=8,wrap=True);fig.savefig(OUT/'figures'/f'{name}.png',dpi=160);pdf.savefig(fig);plt.close(fig);pages.append((name,caption))
  def table(title,cols,rows,name,caption='',size=9):
   fig,ax=plt.subplots(figsize=(14,8.5));ax.axis('off');ax.set_title(title,fontsize=18,pad=25)
   t=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.9)
   for (r,c),cell in t.get_celld().items():cell.set_edgecolor('#cfdbdf');cell.set_facecolor('#dcebee' if r==0 else '#f4f7f8' if r%2 else 'white')
   fig.subplots_adjust(left=.03,right=.97,top=.89,bottom=.12);save(fig,name,caption)
  fig=plt.figure(figsize=(14,8.5));fig.text(.06,.90,'Milestone 1 — traffic actions and GPU work',fontsize=25,weight='bold')
  texts=[f"Frozen simulation commit: {manifest['commit']}", '600 s horizon · 0.25 s timestep · following policy · parallel backend; serial reference', '9 scenarios × 3 unprofiled repeats × 2 backends = 54 full runs.', '27 separate Nsight samples: all six kernels at t = 60, 180, 450 s.', 'Pre-step populations and actions over [t, t + 0.25 s] are recorded for every sample.', 'San Pablo controller comparison: identical routes, lanes, departures and model.', 'Corridor control: 32 independent two-lane straight corridors; unchanged departures.', 'Corridor topology changes road/movement/junction counts and arbitration work; no causal ablation.', 'Pending includes not-yet-departed vehicles; not_departed is separately recorded.', 'Stopped means on-road speed < 0.5 m/s. Connector vehicles are counted separately.', '15 W mode retained; clocks use DVFS. Nsight timing is separate from unprofiled timing.', 'Safety checks and exact serial/repeat state/event comparisons are required for every run.']
  for i,s in enumerate(texts):fig.text(.06,.81-i*.054,s,fontsize=12)
  save(fig,'01-method','Preservation does not imply field calibration. Signals and right-on-red permissions remain synthetic assumptions.')
  a=next(p for p in profiles if p['case']=='controller72_400' and p['time']==180 and p['kernel']=='advance')
  b=next(p for p in profiles if p['case']=='always_permitted_400' and p['time']==180 and p['kernel']=='advance')
  fig=plt.figure(figsize=(14,8.5));fig.text(.06,.90,'What this baseline establishes',fontsize=24,weight='bold')
  observations=[
   'All 54 runs pass the movement/gap audits; serial and parallel outputs agree byte for byte.',
   'At 1,600 requested trips, completed routes are 137 with signals, 441 always-permitted, and 1,600 in the synthetic corridor.',
   'Always-permitted is still congested: 879 trips remain pending at 600 s. Removing red lights does not remove crossing conflicts or overload.',
   'The corridor has no due-vehicle admission backlog and no conflict/space denials; low-speed startup samples are retained.',
   f"At 180 s with 400 trips, moving road vehicles increase from {a['road_moving']:.0f} to {b['road_moving']:.0f}; total road-active changes from {a['road_moving']+a['road_stopped']:.0f} to {b['road_moving']+b['road_stopped']:.0f}.",
   f"Long-scoreboard stalls remain close: {a[METRICS[11]]:.2f}% versus {b[METRICS[11]]:.2f}%. Road propagation duration is {a[METRICS[0]]:.2f} versus {b[METRICS[0]]:.2f} µs.",
   'This counters a simple stopped-traffic explanation for stall percentage. It does not isolate a causal movement effect.',
   'At 450 s, the low-demand and all corridor cases are drained. Their remaining kernel cost reflects empty-network launch/scan overhead, not moving traffic.',
   'At 1,600 trips and t=180 s, corridor road propagation takes 4.92 ms versus 4.25 ms with signals. Yet its full-run GPU time is lower (4.34 versus 6.48 s) because it clears earlier.'
  ]
  y=.80
  for text in observations:
   wrapped=__import__('textwrap').fill(text,125);fig.text(.06,y,wrapped,fontsize=12);y-=.033+.024*len(wrapped.splitlines())
  save(fig,'01-findings','These are measured associations and validation results, not evidence that maximizing GPU utilization improves simulation speed.')
  table('Unprofiled timing: medians of three 600 s runs',['Case','Parallel GPU s','min–max s','Serial GPU s','Advance / junction s','Wall s'],[[s['case'],f"{s['gpu_ms_median']/1000:.3f}",f"{s['gpu_ms_range'][0]/1000:.3f}–{s['gpu_ms_range'][1]/1000:.3f}",f"{s['serial_gpu_ms_median']/1000:.3f}",f"{s['advance_ms_median']/1000:.3f} / {s['junction_ms_median']/1000:.3f}",f"{s['wall_s_median']:.3f}"] for s in summaries],'02-timing','CUDA-event intervals include launch gaps within their ranges. Wall time includes audits and output. GPU counters were collected in separate runs.')
  table('Traffic outcomes and time-weighted activity',['Case','Completed / N','Pending at 600','Mean stopped / moving','Mean pending','Completion / min','Entry / min'],[[s['case'],f"{s['completed']}/{s['n']}",s['pending'],f"{s['time_mean_road_stopped']:.1f} / {s['time_mean_road_moving']:.1f}",f"{s['time_mean_pending']:.1f}",f"{s['completed_per_min']:.1f}",f"{s['entries_per_min']:.1f}"] for s in summaries],'03-outcomes','Means cover the full horizon, including the empty tail after traffic clears. Entry rate counts connector admissions, not new trips.')
  table('Observed regimes and capacity gate',['Case','Observed regime','Max due backlog','Space / conflict denials'],[[r['case'],__import__('textwrap').fill(r['regime'],38),r['maximum_due_backlog_after_spawn'],f"{r['space_denials']} / {r['conflict_or_yield_denials']}"] for r in gate['cases']],'03-regimes','Backlog excludes future departures and is measured after spawning. All movement/gap, conservation and serial-equivalence checks passed.',size=7)
  table('Speed distributions: active vehicle-time samples',['Case','Min','10th percentile','Median','90th percentile','Max','Road stopped %'],[[s['case']]+[f'{v:.2f}' for v in s['speed_quantiles_mps']]+[f"{100*s['road_stopped_sample_fraction']:.1f}"] for s in summaries],'04-speeds','Speeds in m/s; sampled every 0.5 s, road and connector vehicles. Pending/completed excluded. Stopped fraction uses road samples only; spawning at zero speed counts.')
  fig,axes=plt.subplots(1,3,figsize=(14,5))
  for ax,n in zip(axes,(20,400,1600)):
   for mode in MODES:
    speed=np.sort(speed_samples[f'{mode}_{n}']);ax.plot(speed,np.arange(1,len(speed)+1)/len(speed),label=mode)
   ax.set_title(f'{n} trips');ax.set_xlabel('Speed (m/s)');ax.set_ylabel('Cumulative fraction of active samples');ax.legend(fontsize=8);ax.grid(alpha=.2)
  fig.tight_layout(rect=(0,.08,1,1));save(fig,'04-speed-cdf','Vehicle-time weighted: longer waits contribute more samples. Includes road and connector vehicles; excludes pending and completed trips.')
  for ti,t in enumerate(manifest['profile_times']):
   table(f'Parallel road propagation at t = {t} s',['Metric','20: 72 s / permitted / corridor','400: 72 s / permitted / corridor','1,600: 72 s / permitted / corridor'],[[label]+[' / '.join(f"{next(p[k] for p in profiles if p['n']==n and p['mode']==mode and p['time']==t and p['kernel']=='advance'):.3f}" for mode in MODES) for n in (20,400,1600)] for k,label in [(METRICS[i],LABELS[i]) for i in DISPLAY]],f'05-gpu-{t}','One matched timestep per scenario. A drained network still launches threads for pending/completed records. Percentages are per-kernel counters, not whole-device busy time.',size=8)
   table(f'Actions and populations paired with t = {t} s counters',['Case','Pending (future)','Road stop / move','Connector','Completed','Entries / exits','Space / conflict denials'],[[r['case'],f"{r['pending']} ({r['not_departed']})",f"{r['road_stopped']} / {r['road_moving']}",r['connector'],r['completed'],f"{r['entries']} / {r['completed_step']}",f"{r['blocked_space']} / {r['blocked_conflict_or_yield']}"] for r in actionrows if float(r['time'])==t],f'06-actions-{t}','Populations describe the start of the timestep. Actions count the following 0.25 s. Pending + road stopped + road moving + connector + completed = requested.')
  for n in (20,400,1600):
   table(f'Profiled kernel breakdown — {n} trips (µs)',['Case @ time']+manifest['parallel_kernels'],[[f'{mode} @ {t}']+[f"{next(p[METRICS[0]] for p in profiles if p['n']==n and p['mode']==mode and p['time']==t and p['kernel']==kernel):.2f}" for kernel in manifest['parallel_kernels']] for mode in MODES for t in manifest['profile_times']],f'06-breakdown-{n}','Nsight measurements are separate from timing repeats; launch gaps are excluded from individual kernel durations.',size=7)
  for kernel in manifest['parallel_kernels'][1:]:
   table(f'{kernel}: GPU and warp counters at t = 180 s',['Metric','20: 72 s / permitted / corridor','400: 72 s / permitted / corridor','1,600: 72 s / permitted / corridor'],[[label]+[' / '.join(f"{next(p[k] for p in profiles if p['n']==n and p['mode']==mode and p['time']==180 and p['kernel']==kernel):.3f}" for mode in MODES) for n in (20,400,1600)] for k,label in [(METRICS[i],LABELS[i]) for i in DISPLAY]],'07-'+kernel,'All kernels and all three times are available in profile_metrics.csv, including block/grid dimensions.',size=8)
  fig,axes=plt.subplots(3,3,figsize=(14,9),sharex=True)
  for row,n in enumerate((20,400,1600)):
   for col,mode in enumerate(MODES):
    a=axes[row,col];rs=read(ART/'runs'/f'{mode}_{n}'/'parallel_0/step_metrics.csv');x=[float(r['time']) for r in rs]
    for k,color in [('pending','gray'),('road_stopped','#d94d3e'),('road_moving','#168896'),('connector','#c68c20'),('completed','#507047')]:a.plot(x,[float(r[k]) for r in rs],label=k,color=color,lw=1)
    a.set_title(f'{mode}, {n} trips');a.set_xlabel('Simulation seconds');a.set_ylabel('Vehicles')
  axes[0,0].legend(fontsize=7);fig.tight_layout(rect=(0,.05,1,.96));save(fig,'08-populations','Traffic time histories show when a sample represents congestion, startup, movement or an already-drained network.')
  fig,axes=plt.subplots(1,3,figsize=(14,5))
  for mode,color in zip(MODES,['#bb483d','#258494','#72913c']):
   ps=[p for p in profiles if p['kernel']=='advance' and p['mode']==mode]
   for ax,key,label in zip(axes,['road_moving','road_stopped','pending'],['Road moving','Road stopped','Pending']):
    ax.scatter([p[key] for p in ps],[p[METRICS[0]] for p in ps],c=color,label=mode)
    for p in ps:ax.annotate(f"{p['n']}@{p['time']}",(p[key],p[METRICS[0]]),fontsize=6)
    ax.set_xlabel(label);ax.set_ylabel('advance duration (µs)')
  axes[0].legend(fontsize=8);fig.tight_layout(rect=(0,.09,1,1));save(fig,'09-associations','Associations across 27 samples, not causal effects. Population, route geometry, kernel branch paths and inactive padding differ. Milestone 2 isolates these factors.')
  # Descriptive correlations only: fixed N and fixed San Pablo geometry, six windows.
  def ranks(values):
   a=np.asarray(values);return np.array([np.sum(a<v)+(np.sum(a==v)-1)/2 for v in a])
  features=['road_moving','road_stopped','connector','pending','entries','blocked_space','blocked_conflict_or_yield']
  metrics=[METRICS[i] for i in (0,6,7,9,11,1,2)]
  metriclabels=['Duration','SM throughput','Occupancy','Active threads','Long scoreboard','L1 throughput','L2 throughput']
  fig,axes=plt.subplots(1,3,figsize=(14,7));correlations=[]
  for ax,n in zip(axes,(20,400,1600)):
   ps=[p for p in profiles if p['kernel']=='advance' and p['n']==n and p['mode']!='corridor'];matrix=np.full((len(metrics),len(features)),np.nan)
   for i,k in enumerate(metrics):
    for j,feature in enumerate(features):
     a=ranks([p[k] for p in ps]);b=ranks([p[feature] for p in ps])
     if np.std(a)>0 and np.std(b)>0:matrix[i,j]=np.corrcoef(a,b)[0,1]
     correlations.append(dict(trips=n,metric=k,traffic_feature=feature,samples=len(ps),spearman=None if np.isnan(matrix[i,j]) else float(matrix[i,j])))
   im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='coolwarm',aspect='auto');ax.set_title(f'{n} trips: six matched windows');ax.set_xticks(range(len(features)),features,rotation=65,ha='right',fontsize=8);ax.set_yticks(range(len(metrics)),metriclabels,fontsize=9)
   for i in range(len(metrics)):
    for j in range(len(features)):ax.text(j,i,'—' if np.isnan(matrix[i,j]) else f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
  fig.subplots_adjust(left=.09,right=.93,bottom=.30,top=.90,wspace=.5);fig.colorbar(im,ax=axes,shrink=.65,label='Spearman rank correlation');save(fig,'09-correlations','Exploratory only: six samples per panel; no causal attribution or significance claim. Fixed requested N and geometry; active population, signal paths and time still vary. Dash = constant variable.')
  dump(OUT/'descriptive_correlations.json',correlations)
  fig=plt.figure(figsize=(14,8.5));fig.text(.06,.91,'Interpretation and validation boundaries',fontsize=23,weight='bold')
  lines=['advance: per-vehicle motion, leader scans, stop-line and receiving-lane reads.', 'prepare: connector advancement and summary reset; summarize: lane minima and reservations.', 'request: signal eligibility and existing conflicts; arbitrate: priorities and new admissions.', 'spawn: admission of pending vehicles into available road entrances.', 'A zero admission count can still involve substantial scanning and conflict/space checks.',
 'Stopped road vehicles still execute leader scans; inactive records remain in scanned arrays.', 'Long-scoreboard stalls locate a dependency category; they do not identify a particular load.', 'Low occupancy is not proof that stopped vehicles cause low utilization.', 'Compare matched times together with road-active and inactive counts, not trip count alone.', 'Always-permitted retains conflicts, car-following and downstream protection; it is not labeled free-flow.', 'The corridor retains spawning/acceleration and 7 m/s connector speeds. It is not constant-speed travel.', 'All scenario safety audits and serial/repeat equivalence checks passed before inclusion.', 'Hardware counters are single-window samples, not confidence intervals or full-run averages.', 'No Milestone 2 optimizations or constructed snapshot experiments are included.']
  for i,s in enumerate(lines):fig.text(.06,.82-i*.048,s,fontsize=12)
  save(fig,'10-interpretation','Full inputs, binary, commands, manifests, Nsight captures and raw per-step observations are retained in the experiment artifact folder.')
 html=['<!doctype html><meta charset="utf-8"><title>Milestone 1 baseline</title><style>body{max-width:1300px;margin:30px auto;font:18px system-ui}img{width:100%}</style><h1>Milestone 1: traffic actions and GPU work</h1><p><a href="milestone1-baseline.pdf">Download PDF report</a> · <a href="summary.csv">Timing/outcomes CSV</a> · <a href="profile_metrics.csv">All GPU counters</a> · <a href="profile_actions.csv">Matched traffic actions</a> · <a href="manifest.json">Frozen manifest</a></p>']
 for name,caption in pages:html.append(f'<img src="figures/{name}.png"><p>{caption}</p>')
 (OUT/'index.html').write_text('\n'.join(html));print(OUT)
if __name__=='__main__':main()
