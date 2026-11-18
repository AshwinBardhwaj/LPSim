from pathlib import Path
import json,tarfile,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import textwrap
A=Path('artifacts/local-scaling-20261028');R=Path('reports/local-scaling-20261028');R.mkdir(parents=True,exist_ok=True)
archives=list(A.glob('cloud*/*.tar.gz'));runs=[];pop=[];stages=[]
for archive in archives:
 target=archive.parent/'raw';target.mkdir(exist_ok=True)
 with tarfile.open(archive) as t:
  for m in t.getmembers():
   if m.issym() or m.islnk() or not (target/m.name).resolve().is_relative_to(target.resolve()):raise ValueError('Unsafe archive')
  t.extractall(target)
 for directory in target.iterdir():
  if not directory.is_dir():continue
  for file,digest in json.loads((directory/'sha256.json').read_text()).items():
   assert hashlib.sha256((directory/file).read_bytes()).hexdigest()==digest,file
  if not (directory/'runs.json').exists():continue
  for run in json.loads((directory/'runs.json').read_text()):
   row={k:v for k,v in run.items() if k!='metrics'};row.update(run['metrics']);row['label']='dense_single' if row['dense'] else row['policy'];row['compute_ms']=run['metrics'].get('compute_ms',run['metrics'].get('compute_wall_ms'));runs.append(row)
   p=directory/run['path'];df=pd.read_csv(p/'populations.csv');df['case']=row['case'];df['label']=row['label'];df['ranks']=row['ranks'];df['repeat']=row['repeat'];df['verify']=row['verify'];pop.append(df)
   if row['repeat']>0:
    st=pd.read_csv(p/'steps.csv').mean(numeric_only=True).to_dict();st.update(case=row['case'],label=row['label'],ranks=row['ranks'],repeat=row['repeat']);stages.append(st)
T=pd.DataFrame(runs);T.to_csv(R/'all-runs.csv',index=False);P=pd.concat(pop,ignore_index=True);P.to_csv(R/'populations.csv',index=False);S=pd.DataFrame(stages);S.to_csv(R/'stage-means.csv',index=False)
measured=T[(T.repeat>0)&(T.verify==0)];summary=measured.groupby(['case','network','n','label','ranks']).agg(median_ms=('compute_ms','median'),min_ms=('compute_ms','min'),max_ms=('compute_ms','max'),setup_ms=('setup_ms','median'),wire_bytes=('wire_bytes','median'),repeats=('repeat','count')).reset_index()
for i,row in summary.iterrows():
 base=summary[(summary.case==row['case'])&(summary.label=='dense_single')].median_ms.iloc[0];local=summary[(summary.case==row['case'])&(summary.label=='geography')&(summary.ranks==1)].median_ms.iloc[0]
 summary.loc[i,'speedup_dense']=base/row.median_ms;summary.loc[i,'speedup_local1']=local/row.median_ms
 meta=A/'inputs'/row['case']/f"{row['label'] if row['label']!='dense_single' else 'geography'}_{row['ranks']}.json"
 summary.loc[i,'construction_ms']=json.loads(meta.read_text())['construction_ms'] if meta.exists() and row['ranks']>1 else 0
summary['cold_ms']=summary.median_ms+summary.setup_ms+summary.construction_ms;summary.to_csv(R/'timing-summary.csv',index=False)
figures=[]
def save(fig,name):fig.tight_layout();fig.savefig(R/(name+'.png'),dpi=170,bbox_inches='tight');fig.savefig(R/(name+'.pdf'),bbox_inches='tight');figures.append(fig)
fig,axes=plt.subplots(2,3,figsize=(13,8))
for ax,(case,group) in zip(axes.flat,summary.groupby('case',sort=False)):
 for label in ['geography','paper_balanced','paper_community']:
  g=group[group.label==label].sort_values('ranks');ax.plot(g.ranks.to_numpy(),g.median_ms.to_numpy(),'o-',label=label)
 ax.axhline(group[group.label=='dense_single'].median_ms.iloc[0],color='black',ls='--',label='original single');ax.set_title(case);ax.set_xticks([1,2,3]);ax.set_ylabel('Wall ms / 60 steps');ax.set_xlabel('GPU count')
axes[0,0].legend(fontsize=7);save(fig,'01-fixed-workload')
fig,ax=plt.subplots(figsize=(9,5))
for label in ['geography','paper_balanced','paper_community']:
 xs=[];ys=[]
 for k,n in [(1,32768),(2,65536),(3,98304)]:
  selected=summary[(summary.case==f'sf_{n}')&(summary.ranks==k)&(summary.label==(label if k>1 else 'geography'))]
  if len(selected):xs.append(k);ys.append(selected.median_ms.iloc[0])
 ax.plot(xs,ys,'o-',label=label)
ax.set_xlabel('GPUs: initially 32,768 active vehicles per GPU on average');ax.set_ylabel('Wall ms / 60 steps');ax.set_title('Population weak scaling on fixed SF topology');ax.legend();save(fig,'02-population-weak-scaling')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for label in ['geography','paper_balanced','paper_community']:
 g=summary[(summary.ranks==3)&(summary.label==label)].sort_values('n');axes[0].plot(g.n.to_numpy(),g.speedup_dense.to_numpy(),'o-',label=label);axes[1].plot(g.n.to_numpy(),(g.wire_bytes/1e6).to_numpy(),'o-',label=label)
axes[0].axhline(1,color='gray',ls=':');axes[0].set_ylabel('Speedup relative to original single GPU');axes[1].set_ylabel('Wire MB / 60 steps')
for ax in axes:ax.set_xlabel('Initial active vehicles');ax.legend(fontsize=8)
save(fig,'03-speedup-and-messages')
# Authority counts are separate from required ghosts and spare pool capacity.
last=P[(P.repeat==1)&(P.label!='dense_single')].copy();last=last.sort_values('step').groupby(['case','label','ranks','rank']).tail(1);last.to_csv(R/'final-local-populations.csv',index=False)
# Largest-case stage and allocation views expose where local processing still costs time.
largest=summary.loc[summary.n.idxmax(),'case'];fig,ax=plt.subplots(figsize=(11,5));bottom=np.zeros(6)
configs=[(p,k) for k in (2,3) for p in ('geography','paper_balanced','paper_community')]
for col,label in [('motion_ms','Motion/indexes'),('exchange0_ms','State exchange 0'),('junction_ms','Junction'),('reservation_ms','Reservations'),('spawn_ms','Spawn'),('exchange1_ms','State exchange 1')]:
 values=np.array([S[(S.case==largest)&(S.label==p)&(S.ranks==k)][col].mean() for p,k in configs]);ax.bar(np.arange(6),values,bottom=bottom,label=label);bottom+=values
ax.set_xticks(np.arange(6),[f'{p} / {k}' for p,k in configs],rotation=20,ha='right');ax.set_ylabel('Mean wall ms / step');ax.set_title(largest+' — local processing stage costs');ax.legend(fontsize=8);save(fig,'04-stage-breakdown')
allocation=[]
for (case,label,k),g in last.groupby(['case','label','ranks']):
 active=g.stopped+g.moving+g.connector
 allocation.append(dict(case=case,policy=label,ranks=k,active=int(active.sum()),active_min=int(active.min()),active_max=int(active.max()),active_max_mean=float(active.max()/max(1,active.mean())),ghosts=int(g.ghosts.sum()),processed_records=int(g.processed_records.sum()),reserved_capacity=int(g.reserved_capacity.sum())))
alloc=pd.DataFrame(allocation);alloc.to_csv(R/'allocation-summary.csv',index=False)
# Read explicit cross-date billing, never reset the allowance at UTC midnight.
billing=[]
for p in sorted(A.glob('billing*.json'),key=lambda p:p.stat().st_mtime):
 d=json.loads(p.read_text())
 if isinstance(d,list) and d:billing=d
cost=sum(float(x['cost']) for x in billing if x.get('description') in ['lpsim-a100-partition-study','lpsim-local-scaling'])
budget=json.loads((A/'budget.json').read_text())
info=dict(budget_reserved_upper_usd=budget['total_reserved_upper_usd'],cases=int(summary.case.nunique()),timing_runs=len(measured),verified_runs=int((T.verify==1).sum()),verified_steps=int(T[T.verify==1].steps.sum()),cost_usd=cost,max_speedup_dense=float(summary[summary.ranks>1].speedup_dense.max()),min_speedup_dense=float(summary[summary.ranks>1].speedup_dense.min()))
(R/'findings.json').write_text(json.dumps(info,indent=2))
with PdfPages(R/'local-scaling-report.pdf') as pdf:
 def page(title,paragraphs):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,title,fontsize=19,weight='bold');y=.85
  for p in paragraphs:
   lines=textwrap.wrap(p,112);fig.text(.06,y,'\n'.join(lines),fontsize=11,va='top',linespacing=1.4);y-=.028*len(lines)+.03
  fig.text(.06,.035,'LPSim • local processing and paper-style partition baselines • 2026-10-28',fontsize=9);pdf.savefig(fig);plt.close(fig)
 page('Paper-style partitioning and local vehicle processing',[
  'Question: can traffic-weighted partitions outperform geography, and does more actual work overcome multi-GPU overhead when vehicle processing is local?',
  f"Executed {info['cases']} cases and {info['timing_runs']} measured repetitions. Exact reference comparisons cover {info['verified_runs']} runs / {info['verified_steps']} timesteps. Per-case policy and GPU-count final hashes must agree. Full raw evidence is archived.",
  'Policies share identical physics and memory structures. METIS balances visited node counts while minimizing route-weighted cuts. Leiden modularity communities are grouped geographically using k-means. Both use known route demand, not future simulated occupancy. These are paper-style baselines, not an exact reproduction of unspecified original parameters.',
  'Each GPU processes compact owned-plus-ghost records in ascending stable-ID order. Leader and request indexing now use local record counts. Full records are exchanged and compact lists rebuilt each phase; those costs are included. Network-sized summaries remain replicated. Persistent pools have initial-local-count headroom and fail safely on overflow; online growth is not implemented.',
  f"Cumulative posted Modal usage: USD {cost:.3f}; current-hour billing may lag. Prior posted usage plus this job's full reserved timeout is USD {budget['total_reserved_upper_usd']:.3f}, below USD 10. The figures describe this budgeted pilot, not a learned-policy result." ])
 page('Measured findings and counter-evidence',[
  'The largest test ends with 98,244 active vehicles out of 98,304 initially active; only 60 completed. Insufficient admitted population is therefore not the explanation for this result.',
  'At that size, the original one-GPU continuation takes 29.313 ms. Local one-GPU takes 47.535 ms. The best two-GPU median is 185.769 ms (community); the best three-GPU median is 216.028 ms (balanced). Local storage is correct, but the new execution framework does not achieve a net multi-GPU speedup.',
  'There is a policy effect: two-GPU community partitioning is 11.0% faster than two-GPU geography in this case, before construction/setup costs. Three-GPU balanced partitioning is 4.9% faster than geography. Only three repetitions were measured; inspect their ranges and do not present these as held-out or statistically established research gains.',
  'For three GPUs, geography transfers 110.906 MB during the continuation, versus 20.128 MB for balanced and 11.267 MB for community. Balanced also reduces final active max/mean from 1.628 to 1.124. Community cuts communication further but has active max/mean 1.473. Thus lower bytes, lower imbalance and lower runtime are different objectives.',
  'State exchange plus reservation stages occupy about 73% of the largest balanced three-GPU step. These include retaining local records, sorting stable IDs, full-record packing/copies, host count handling and barriers. This is not evidence that NVLink bandwidth itself is saturated. Next, subdivide these stages before changing the allocator.',
  'Population weak scaling also worsens: the local one-GPU 32,768-active case takes 38.289 ms; the balanced two-GPU 65,536 case takes 154.880 ms; balanced three-GPU 98,304 takes 216.028 ms. No scaling crossover is demonstrated in the tested range.' ])
 page('Workload scope and fair interpretation',[
  'The converted SF network has 39,269 roads, 93,612 movements and 17,348 junctions. Berkeley provides a smaller real topology. Cached geometry is approximate and signal/turn controls synthetic. One SF road with 11 source lanes is clamped to the supported maximum of eight.',
  'Scaling snapshots are constructed, safely spaced initial road states: 15 m spacing, 5 m/s initial speed, seeded stochastic lane-continuous routes with a modest main-road bias. All requested vehicles are initially active. They are performance workloads, not calibrated traffic trajectories. Historical simulated Berkeley queues/connectors provide an additional correctness case.',
  'Fixed-workload scaling compares identical states on one, two and three GPUs. The weak-scaling panel increases initial active population from 32,768 to 65,536 to 98,304 on one, two and three GPUs. The SF topology stays fixed: this is population weak scaling, not simultaneous network-and-population weak scaling.',
  'Unprofiled timings cover 60 original 0.25-second steps (15 simulated seconds), with one warm-up and three measured repetitions. Exact validation uses separate 20-step runs. Initial and final owned/ghost/active counts distinguish shrinking workloads from speedup. Longer 600-second runs remain a subsequent validation requirement.',
  'Original single-GPU timing retains the optimized prior execution path. Local one-GPU timing measures the new framework overhead. Report both references; do not credit a partition policy for changes to execution infrastructure. No new Nsight counters are collected in this budgeted pilot; previous backend counters cannot be relabeled as these results.' ])
 for fig in figures:pdf.savefig(fig)
 for case,g in summary.groupby('case',sort=False):
  df=g[['label','ranks','median_ms','min_ms','max_ms','speedup_dense','setup_ms','construction_ms','wire_bytes']].copy();df.wire_bytes/=1e6;df.columns=['Policy','GPUs','Median ms','Min ms','Max ms','vs original','Setup ms','Partition ms','Wire MB']
  for c in df.columns[2:]:df[c]=df[c].map(lambda v:f'{v:.3f}')
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(case,fontsize=18);tb=ax.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='center',colWidths=[.22,.06]+[.10]*7);tb.auto_set_font_size(False);tb.set_fontsize(9);tb.scale(1,2);pdf.savefig(fig);plt.close(fig)
 df=alloc[alloc.case==largest][['policy','ranks','active','active_min','active_max','ghosts','processed_records','reserved_capacity']].copy();df.columns=['Policy','GPUs','Active total','Min/GPU','Max/GPU','Ghosts','Processed','Capacity']
 fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(largest+' — final actual allocation',fontsize=18);tb=ax.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='center',colWidths=[.23]+[.105]*7);tb.auto_set_font_size(False);tb.set_fontsize(9);tb.scale(1,2);fig.text(.07,.10,'Counts at the end of the 15-second continuation; spare capacity is not processed work.',fontsize=10);pdf.savefig(fig);plt.close(fig)
 page('What this experiment can establish',[
  'A speedup must exceed timing variability and persist after charging partition construction and setup. Construction includes common graph/demand preparation and is measured on the Jetson host; one GPU requires no partition construction and is charged zero. This is recorded separately; it is not a measured A100 host construction benchmark. It should be remeasured in deployment before a net performance claim.',
  'If the local path remains slower, inspect full-record exchange, compaction/sort passes, network-wide summary work and host synchronization before attributing the result to insufficient active population. Larger workloads test amortization but do not guarantee a crossover.',
  'If a paper-style partition wins over geography, that validates a stronger baseline, not our own novel predictor. A later traffic prior must improve on both methods with equal information access and on held-out demand, including prediction, construction and migration costs.',
  'References: https://arxiv.org/html/2406.08496v1 (Section 3.3); preserved local PROTOCOL.md; timing-summary.csv, final-local-populations.csv, stage-means.csv and raw cloud archives. No forecast or generalization result is claimed.' ])
for f in figures:plt.close(f)
print(json.dumps(info,indent=2))
