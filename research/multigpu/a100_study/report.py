from pathlib import Path
import json,textwrap,re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path('reports/a100-study-20261028');A=Path('artifacts/a100-study-20261028');R.mkdir(exist_ok=True)
T=pd.read_csv(R/'timing-summary.csv');D=pd.read_csv(R/'diagnostic-windows.csv');P=pd.read_csv(R/'populations.csv');L=pd.read_csv(R/'allocations.csv')
labels=['single','geography','vehicle_count','communication_graph'];names={'single':'1 A100','geography':'3 A100: geography','vehicle_count':'3 A100: active count','communication_graph':'3 A100: graph'};colors=['#354f60','#258e8a','#e49733','#875fa0']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
figures=[]
def save(fig,name):
 fig.savefig(R/(name+'.png'),bbox_inches='tight',dpi=180);fig.savefig(R/(name+'.pdf'),bbox_inches='tight');figures.append((name,fig))
# Comparison figures show complete continuation time, not a selected kernel.
fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
for ax,n in zip(axes,[2048,8192,16384]):
 for i,label in enumerate(labels):
  g=T[(T.n==n)&(T.label==label)].set_index('pattern').reindex(['distributed','concentrated','bursty'])
  if g.median_ms.isna().all():continue
  x=np.arange(3)+(i-1.5)*.19;ax.bar(x,g.median_ms/1000,.18,color=colors[i],label=names[label]);ax.errorbar(x,g.median_ms/1000,yerr=np.vstack([(g.median_ms-g.ci_low_ms)/1000,(g.ci_high_ms-g.median_ms)/1000]),fmt='none',ecolor='#222',capsize=2)
 ax.set_xticks(range(3),['Distributed','Concentrated','Bursty'],rotation=20);ax.set_title(f'{n:,} requested trips');ax.set_ylabel('180–600 s continuation wall time (s)');ax.set_ylim(bottom=0)
axes[-1].legend(fontsize=8);fig.suptitle('Baseline versus partition policies — five unprofiled repeats');fig.tight_layout();save(fig,'01-runtime')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,n in zip(axes,[2048,8192,16384]):
 g=D[(D.n==n)&(D.label!='single')].groupby('label').mean(numeric_only=True).reindex(labels[1:]);bottom=np.zeros(3)
 for col,color,title in [('motion_ms','#387898','Motion + indexes'),('junction_ms','#5fa8a1','Requests + arbitration'),('spawn_ms','#a8c6aa','Spawn'),('exchange0_ms','#e7ae61','State exchange 0'),('reservation_ms','#bf775d','Reservation exchange'),('exchange1_ms','#956c98','State exchange 1')]:
  ax.bar(range(3),g[col],bottom=bottom,color=color,label=title);bottom+=g[col].fillna(0).to_numpy()
 ax.set_xticks(range(3),['Geography','Count','Graph'],rotation=20);ax.set_title(f'{n:,} trips');ax.set_ylabel('Mean wall ms/step (diagnostic windows)')
axes[-1].legend(fontsize=7);fig.suptitle('Where time goes: communication stages include pack, waits and apply');fig.tight_layout();save(fig,'02-stage-breakdown')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,n in zip(axes,[2048,8192,16384]):
 g=T[(T.n==n)&(T.label!='single')].groupby('label').mean(numeric_only=True).reindex(labels[1:]);ax.bar(range(3),g.wire_bytes_median/1e9,color=colors[1:]);ax.set_xticks(range(3),['Geography','Count','Graph'],rotation=20);ax.set_title(f'{n:,} trips');ax.set_ylabel('State + reservation transfer GB / continuation')
fig.suptitle('Partition policy changes real boundary traffic');fig.tight_layout();save(fig,'03-boundary-bytes')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for i,label in enumerate(labels[1:]):
 g=D[D.label==label].groupby('n').mean(numeric_only=True);axes[0].plot(g.index.to_numpy(),g.active_max_mean.to_numpy(),'o-',color=colors[i+1],label=names[label]);axes[1].plot(g.index.to_numpy(),g.balance_only_ideal_pct.to_numpy(),'o-',color=colors[i+1],label=names[label])
axes[0].axhline(1,color='gray',ls=':');axes[0].set_ylabel('Active population: max GPU / mean GPU');axes[1].set_ylabel('Ideal balance-only savings / step (%)')
for ax in axes:ax.set_xlabel('Requested trips');ax.legend(fontsize=8)
fig.suptitle('Balancing population is different from removing the measured critical path');fig.tight_layout();save(fig,'04-balance-opportunity')
# Time evolution of actual active population and ghosts, for the largest available workload.
n=int(T.n.max());case=f'train_concentrated_{n}';fig,axes=plt.subplots(3,1,figsize=(11,7),sharex=True)
for ax,label in zip(axes,labels[1:]):
 g=P[(P.case==case)&(P.label==label)].copy();g['active']=g.stopped+g.moving+g.connector
 for rank,h in g.groupby('rank'):ax.plot(h.time.to_numpy(),h.active.to_numpy(),label=f'GPU {rank}',linewidth=1)
 ax.set_title(names[label]);ax.set_ylabel('Owned active vehicles');ax.legend(ncol=3,fontsize=8)
axes[-1].set_xlabel('Simulation time (s)');fig.suptitle(f'Actual workload distribution: concentrated {n:,}-trip demand');fig.tight_layout();save(fig,'05-active-populations')
# Draw the actual junction-owner maps used in the medium distributed case.
from matplotlib.collections import LineCollection
geo=json.loads(Path('artifacts/multigpu-preparation-20261027/inputs/network/geometry.json').read_text())
paths=list(A.glob('*/raw/*/train_distributed_8192__allocation'))
if paths:
 fig,axes=plt.subplots(1,3,figsize=(12,6))
 roads=geo['roads'];segments=[[r['start'],r['end']] for r in roads]
 for ax,label in zip(axes,labels[1:]):
  owners=list(map(int,(paths[-1]/(label+'.txt')).read_text().split()))
  ax.add_collection(LineCollection(segments,colors=[colors[1+owners[r['b']]] for r in roads],linewidths=.45));ax.autoscale();ax.set_aspect('equal');ax.set_title(names[label]);ax.set_axis_off()
 fig.suptitle('Actual ownership maps at 180 s: same Berkeley network and traffic state');fig.tight_layout();save(fig,'07-partition-maps')
# Summary tables for exact numbers.
T['seconds']=T.median_ms/1000
pivot=T.pivot(index=['n','pattern'],columns='label',values='seconds').reindex(columns=labels)
pivot.to_csv(R/'runtime-table.csv')
N=pd.DataFrame()
try:N=pd.read_csv(R/'nsight-raw.csv')
except pd.errors.EmptyDataError:pass
ns=[]
if not N.empty:
 N=N[N.exit_code==0].copy()
 N['value']=pd.to_numeric(N['Metric Value'].astype(str).str.replace(',','',regex=False),errors='coerce')
 time_rows=N['Metric Name']=='gpu__time_duration.sum'
 for unit,factor in [('ns',.001),('us',1.),('µs',1.),('ms',1000.),('s',1e6),('nsecond',.001),('usecond',1.),('msecond',1000.),('second',1e6)]:
  mask=time_rows&(N['Metric Unit']==unit);N.loc[mask,'value']*=factor;N.loc[mask,'Metric Unit']='microseconds'
 assert set(N.loc[time_rows,'Metric Unit'])=={'microseconds'}, 'Unknown Nsight duration units'
 N['kernel']=N['Kernel Name'].astype(str).map(lambda x:next((k for k in ['advanceCompact','arbitrateIndexedJunctions','makeLeaderKeys','linkLaneLeaders'] if k in x),x))
 N['label']=np.where(N.backend=='single','single',N.policy)
 ns=N.groupby(['case','time','label','kernel','Metric Name','Metric Unit']).value.agg(['mean','min','max','count']).reset_index();ns.to_csv(R/'nsight-summary.csv',index=False)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,metric,title in zip(axes,['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','sm__warps_active.avg.pct_of_peak_sustained_active','sm__throughput.avg.pct_of_peak_sustained_elapsed'],['Long-scoreboard stall (%)','Achieved warp occupancy (%)','SM throughput (% peak)']):
  g=N[(N.kernel=='advanceCompact')&(N['Metric Name']==metric)].groupby('label').value.mean().reindex(labels)
  ax.bar(range(4),g,color=colors);ax.set_xticks(range(4),['Single','Geography','Count','Graph'],rotation=20);ax.set_ylabel(title)
 fig.suptitle('Nsight Compute: motion kernel, sampled windows/devices (unweighted mean)');fig.tight_layout();save(fig,'06-nsight-motion')
 popns=pd.read_csv(R/'nsight-populations.csv');popns['label']=np.where(popns.backend=='single','single',popns.policy);popns['active']=popns.stopped+popns.moving+popns.connector
 wide=N[N.kernel=='advanceCompact'].pivot_table(index=['case','time','label','Device'],columns='Metric Name',values='value').reset_index();wide['rank']=pd.to_numeric(wide['Device']);joined=wide.merge(popns,on=['case','time','label','rank'],how='left');joined.to_csv(R/'nsight-state-matched.csv',index=False)
 fig,axes=plt.subplots(1,2,figsize=(12,4))
 for i,label in enumerate(labels):
  g=joined[joined.label==label];axes[0].scatter(g.active,g['gpu__time_duration.sum'],label=names[label],color=colors[i],alpha=.8);axes[1].scatter(g.active,g['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'],label=names[label],color=colors[i],alpha=.8)
 axes[0].set_ylabel('Motion kernel duration (microseconds)');axes[1].set_ylabel('Long-scoreboard stalls (% active warp cycles)')
 for ax in axes:ax.set_xlabel('Owned road + connector vehicles on sampled GPU');ax.legend(fontsize=7)
 fig.suptitle('Matched traffic states and motion counters — association, not a causal ablation');fig.tight_layout();save(fig,'08-state-counter-association')

# Actual GPU model may vary between Modal allocations; preserve this distinction.
hardware=[]
for p in A.glob('*/raw/*/hardware.txt'):
 text=p.read_text();models=re.findall(r'Product Name\s+: (.*)',text);uuids=re.findall(r'GPU UUID\s+: (.*)',text)
 hardware.append({'run':p.parent.name,'models':'; '.join(models),'uuids':'; '.join(uuids)})
pd.DataFrame(hardware).to_csv(R/'hardware.csv',index=False)
# All numerical claims are derived below from saved data.
ratios=[]
for case,g in T.groupby('case'):
 g=g.set_index('label');
 if 'single' in g.index:
  for label in labels[1:]:
   if label in g.index:ratios.append(dict(case=case,label=label,slowdown=g.loc[label,'median_ms']/g.loc['single','median_ms']))
ratio=pd.DataFrame(ratios);ratio.to_csv(R/'relative-runtime.csv',index=False)
billing=[]
for p in sorted(A.glob('billing*.json'),key=lambda x:x.stat().st_mtime):
 try:
  r=json.loads(p.read_text())
  if isinstance(r,list):billing=r
 except Exception:pass
cost=sum(float(r['cost']) for r in billing if r.get('description')=='lpsim-a100-partition-study')
summary={'cases':int(T.case.nunique()),'timing_repetitions':int(T.repeats.sum()),'peer_slowdown_min':float(ratio.slowdown.min()),'peer_slowdown_max':float(ratio.slowdown.max()),'balance_screen_max_percent':float(D.balance_only_ideal_pct.max()),'posted_study_cost_usd':cost,'mean_policy_seconds':T.groupby('label').seconds.mean().to_dict()}
peerD=D[D.label!='single'].copy();peerD['exchange_fraction']=(peerD.exchange0_ms+peerD.reservation_ms+peerD.exchange1_ms)/peerD.total_ms
summary['exchange_fraction_mean']=float(peerD.exchange_fraction.mean());summary['exchange_fraction_range']=[float(peerD.exchange_fraction.min()),float(peerD.exchange_fraction.max())]
summary['balance_screen_mean_percent']=float(peerD.balance_only_ideal_pct.mean())
oracleData=pd.read_csv(R/'restricted-oracle.csv');summary['restricted_oracle_choices']=oracleData.selected_policy.value_counts().to_dict()
(R/'findings.json').write_text(json.dumps(summary,indent=2))
# PDF pages: motivation first, figures and exact data next, limits and next steps last.
with PdfPages(R/'a100-partition-study.pdf') as pdf:
 def textpage(title,paragraphs):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,title,fontsize=20,weight='bold');y=.85
  for para in paragraphs:
   lines=textwrap.wrap(para.replace('$','USD '),112,break_long_words=False);fig.text(.06,y,'\n'.join(lines),fontsize=11,va='top',linespacing=1.5);y-=.029*len(lines)+.035
  fig.text(.06,.035,'LPSim • October 28, 2026 • Measured A100 baseline; no learned-prior claim',fontsize=9,color='#555');pdf.savefig(fig);plt.close(fig)
 def tablepage(title,df,notes):
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,fontsize=17,pad=30);df=df.copy().fillna('—');table=ax.table(cellText=df.values,colLabels=df.columns,colWidths=([.34]+[.165]*4) if list(df.columns)==['Metric','Single','Geography','Count','Graph'] else None,loc='upper center',cellLoc='center',bbox=[0,.18,1,.72]);table.auto_set_font_size(False);table.set_fontsize(9)
  for (r,c),cell in table.get_celld().items():cell.set_edgecolor('#d2dde1');cell.set_facecolor('#dcebf0' if r==0 else ('#f1f5f6' if r%2==0 else 'white'))
  fig.text(.07,.09,'\n'.join(textwrap.wrap(notes,118)),fontsize=10);pdf.savefig(fig);plt.close(fig)
 textpage('From traffic priors to a measurable multi-GPU baseline',[
  'Research question: can network structure and anticipated demand choose partitions that reduce complete simulation runtime, after charging computation, boundary communication, ownership migration and allocation overhead?',
  'Motivation: earlier single-GPU scheduling tests offered limited recoverable imbalance. Multiple GPUs introduce explicit ownership boundaries and communication, giving a network-aware policy a different decision to make. This study establishes that mechanism before fitting a predictor.',
  f'Executed: original physical-device correctness; concurrent direct-peer implementation and correctness; {summary["cases"]} development workload comparisons with {summary["timing_repetitions"]} measured repetitions; separate population/stage diagnostics and Nsight Compute sampling.',
  f'Headline: the tested three-GPU continuations take {summary["peer_slowdown_min"]:.2f}–{summary["peer_slowdown_max"]:.2f} times the single-GPU time. A reduction in boundary traffic is not automatically a net multi-GPU speedup. These are results for the present conservative, dense-slot backend.',
  f'Budget: Modal currently reports ${cost:.3f} for this study, including failed attempts, against a $10 authorization. See billing JSON and stopped-app records; posted usage is not the final invoice.' ])
 textpage('1. Correctness before performance',[
  'Why: a faster partition policy is meaningless if it changes traffic physics, priority order, receiving-space reservations, or right-on-red decisions.',
  'Original backend: three policies × logical/physical devices × 120 timesteps = 720 timestep comparisons. Concurrent backend: six populated-snapshot runs × 120 steps plus three fresh 600-second runs × 2,400 steps = 7,920 timestep comparisons. Every comparison passed exact vehicle-state, ownership/ghost and road/connector/conflict checks.',
  'The concurrent implementation uses three persistent CPU launch workers, one GPU per partition, destination-packed peer state messages, compact broadcast ownership changes, and authoritative reservation messages. It keeps phase barriers and the original bulk-radix leader index, compact motion and indexed junction arbitration.',
  'Ownership is by complete junction and incoming roads. A vehicle crossing an ownership boundary changes its authoritative GPU. Receiving-road vehicles are ghosted to the upstream junction owner. Dense stable-ID vehicle slots and immutable network/routes remain replicated on every GPU.',
  'Frozen source/input bundle: artifacts/multigpu-preparation-20261027. New executable and transport: research/multigpu/a100_study/concurrent.cu. Gate archives and SHA-256 manifests are in artifacts/a100-study-20261028.' ])
 textpage('2. A fair policy comparison',[
  'Why: count balancing at an empty initial state falls back to unit junction weights. To measure actual vehicle-count balancing, each fresh training demand runs to 180 seconds on one GPU. All policies see exactly that same observed state and remain fixed through 600 seconds.',
  'Primary timing is the full 180–600 second continuation (1,680 original 0.25-second steps). It is not a repeated 0–600 second wall-time result. Common prefix, assignment construction, initialization and process/serialization costs are saved separately. Timed stage sums include all required packing, transfers and synchronization.',
  'Three requested populations × three departure patterns: 2,048 / 8,192 / 16,384 trips; distributed / concentrated / bursty. Synthetic routes favor lane-count/speed proxies modestly while retaining randomness. Actual active, pending, stopped, moving, connector and completed populations are recorded; requested demand is not active workload.',
  'Policies: equal-width geography; observed active-count greedy balancing; geography-initialized communication graph refinement with a population-balance penalty. The graph edge weight is a static lane-based byte proxy, not a learned traffic forecast or an optimal partition.',
  'One warm-up plus five timing repetitions, randomized policy order (2710), separate instrumented runs, identical final-state hashes. All policies for a population run on the same allocation. GPU clocks are not locked; actual device identity and telemetry are preserved. These are development cases, not the preregistered held-out success evaluation.' ])
 tablepage('Actual hardware by allocation',pd.DataFrame([{'Session':h['run'].replace('study-',''),'Actual GPUs':str(len(h['models'].split('; ')))+' x '+h['models'].split('; ')[0].replace('NVIDIA ','')} for h in hardware]),'Modal may upgrade a 40GB request to 80GB hardware. Within-case policy comparisons share an allocation. Cross-population or profiler comparisons across different models must not be interpreted as controlled hardware scaling. Full UUIDs and hardware logs are preserved.')
 textpage('Interconnect evidence and its limits',[
  'All three timed population matrices and the profiler allocation reported three NVIDIA A100-SXM4-80GB devices. The final 2,048-trip allocation additionally reported twelve enabled NVLink links per GPU, each listed at 25 GB/s by nvidia-smi nvlink --status. This is link-status evidence, not an application bandwidth measurement.',
  'The container topology query nvidia-smi topo -m was unavailable. We therefore do not claim an independently verified NVSwitch wiring diagram or a specific hop topology for these allocations. Hardware UUIDs and raw command outputs are archived.',
  'The peer backend uses CUDA peer transfers between three devices in one container. These are intra-host GPU exchanges, not inter-cluster network messages. Transfer-stage timings include packing, host count reads, barriers and applying updates; frequent short messages can be expensive even with NVLink available.' ])
 for name,fig in figures:pdf.savefig(fig,bbox_inches='tight')
 df=pivot.reset_index();df.columns=['Trips','Demand','1 GPU (s)','Geography (s)','Count (s)','Graph (s)'];
 for c in df.columns[2:]:df[c]=df[c].map(lambda x:f'{x:.3f}')
 tablepage('Repeated continuation runtime: exact medians',df,'Five unprofiled repetitions; one warm-up excluded. Error bars in the figure are bootstrap confidence intervals of the median across five repetitions, not evidence of cross-network generalization. All comparisons use identical physical states and calculations.')
 cold=T.pivot(index=['n','pattern'],columns='label',values='cold_median_ms').reindex(columns=labels)/1000;df=cold.reset_index();df.columns=['Trips','Demand','1 GPU (s)','Geography (s)','Count (s)','Graph (s)']
 for c in df.columns[2:]:df[c]=df[c].map(lambda x:f'{x:.3f}')
 tablepage('Cold costs: continuation + GPU setup + assignment',df,'Assignment construction is charged once per continuation. GPU context/buffer setup is included here. File input, output serialization and the common 0–180 s prefix are separate; see per-run process wall time and prefix records. These figures are not an end-to-end deployment billing estimate.')
 if isinstance(ns,pd.DataFrame) and not ns.empty:
  metric_names={'gpu__time_duration.sum':'Kernel duration (µs)','sm__throughput.avg.pct_of_peak_sustained_elapsed':'SM throughput (% peak)','sm__warps_active.avg.pct_of_peak_sustained_active':'Achieved occupancy (%)','smsp__warps_eligible.avg.per_cycle_active':'Eligible warps/cycle','smsp__thread_inst_executed_per_inst_executed.ratio':'Active threads/instruction','smsp__issue_active.avg.pct_of_peak_sustained_active':'Issue activity (%)','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct':'Long-scoreboard stall (%)','smsp__warp_issue_stalled_wait_per_warp_active.pct':'Fixed wait stall (%)','l1tex__throughput.avg.pct_of_peak_sustained_elapsed':'L1/TEX throughput (%)','lts__throughput.avg.pct_of_peak_sustained_elapsed':'L2 throughput (%)','launch__registers_per_thread':'Registers/thread'}
  for case in N.case.unique():
   for kernel in ['advanceCompact','arbitrateIndexedJunctions','makeLeaderKeys','linkLaneLeaders']:
    data=N[(N.kernel==kernel)&(N.case==case)].groupby(['Metric Name','label']).value.mean().unstack().reindex(index=list(metric_names),columns=labels);data.index=[metric_names[x] for x in data.index];data=data.map(lambda x:f'{x:.3f}' if pd.notna(x) else '—') if hasattr(data,'map') else data.applymap(lambda x:f'{x:.3f}' if pd.notna(x) else '—');data=data.reset_index();data.columns=['Metric','Single','Geography','Count','Graph'];tablepage('Nsight: '+case.replace('train_','')+' / '+kernel,data,'Unweighted means across 180/360/540 s and sampled GPU kernels for this workload only. Exact per-device/time values and units are in nsight-raw.csv and nsight-summary.csv. Replay is separate from performance timing. Kernel occupancy/stall percentages do not measure complete-run idle time.')
 for n in sorted(T.n.unique()):
  data=[]
  for _,row in T[(T.n==n)&(T.label!='single')].iterrows():
   ar=L[(L.case==row['case'])&(L.policy==row['label'])]
   dd=D[(D.case==row['case'])&(D.label==row['label'])]
   data.append({'Demand':row['pattern'],'Policy':row['label'].replace('communication_graph','graph').replace('vehicle_count','count'),'Cut roads':int(ar.cut_roads.iloc[0]) if len(ar) else '—','Migrations':int(row.migrations_median),'Wire MB':f'{row.wire_bytes_median/1e6:.1f}','Owner ctrl MB':f'{row.control_bytes_median/1e6:.2f}','Active max/mean':f'{dd.active_max_mean.mean():.2f}','Ghosts (mean)':f'{dd.mean_ghosts.mean():.0f}'})
  tablepage(f'Allocation accounting: {int(n):,} requested trips',pd.DataFrame(data),'Wire bytes are actual inter-device state plus reservation payloads over 180–600 s. Compact ownership-control broadcasts are separate. Population/ghost means come from matched 10-second diagnostic windows. All GPUs retain N allocated stable-ID slots, regardless of owned population.')
 memory=pd.read_csv(R/'memory.csv')
 memory=memory[memory['repeat'].between(1,5)]
 memory['n']=memory.case.str.extract(r'_(\d+)$')[0].astype(int)
 mt=memory.groupby(['n','label'])[['vehicle_slots','car_arrays_bytes','compact_state_bytes','peer_state_pool_bytes','owner_delta_pool_bytes','reservation_pool_bytes']].median().reset_index()
 table=[]
 for _,row in mt.iterrows():
  table.append({'Trips':int(row['n']),'Policy':row['label'].replace('communication_graph','graph').replace('vehicle_count','count'),'Slots/GPU':int(row.vehicle_slots),'Car arrays MB/GPU':f'{row.car_arrays_bytes/1e6:.2f}','Peer state pools MB/GPU':f'{row.peer_state_pool_bytes/1e6:.2f}','Owner + reservation MB/GPU':f'{(row.owner_delta_pool_bytes+row.reservation_pool_bytes)/1e6:.2f}'})
 tablepage('Memory allocation remains dense and policy-independent',pd.DataFrame(table),'These are explicitly accounted major allocations, not total memory consumption. Original buffers, network/routes, index scratch and CUDA context overhead also exist; raw memory.csv records device free/total memory. All three policies reserve the same capacities for a given requested population. This study does not yet implement a memory-budget allocator.')
 textpage('What the counters say about the current processing layout',[
  'Motion kernels launch 128 threads per block over the full requested-ID array on every GPU: 64 blocks for 8,192 records and 128 blocks for 16,384 records. Each A100 has 108 SMs. Splitting ownership does not reduce this launch domain or create enough additional independent work to fill the device.',
  'The source-level mechanism is explicit: advanceCompact checks the owner map, but non-owned records still execute writeCompact(next[i], old.load(i)) before returning. extractCompact and bulk leader sorting also traverse dense arrays. This is an implementation property, not a calibrated relationship between congestion and GPU cost.',
  'Across the sampled medium/large states, motion lasts about 12–13 microseconds per GPU, with roughly 6% achieved occupancy and 59–62% long-scoreboard stalls. Active threads per instruction average about 26 on the single GPU and about 12 in partitioned runs. Consult the per-workload tables for exact values and variation.',
  'Junction arbitration is only about 20–24 microseconds per sampled kernel, although its long-scoreboard percentage is about 82–83%. A high stall percentage on a short kernel does not establish that it dominates the complete step. The separate stage measurements include index/sort launches and host coordination that these four selected kernel profiles do not cover.',
  'The next baseline correction should make per-device processing proportional to owned-plus-required-ghost work, preserve stable priority IDs, and measure launch/coordination overhead with a timeline. Larger networks and genuinely larger active populations should then test the scaling crossover. Increasing requested trips that remain pending is not sufficient.' ])
 textpage('3. What allocation can and cannot recover',[
  'Why measure the critical path: balancing active vehicles can increase boundary cuts and ghost populations. A policy must minimize the slowest partition plus unhidden communication and coordination, not merely equalize vehicle counts.',
  f'The balance-only opportunity screen assumes each measured motion and junction stage could run at the mean device duration instead of its maximum, with transport unchanged. The largest sampled saving is {summary["balance_screen_max_percent"]:.2f}% of diagnostic step time. This is an optimistic arithmetic screen, not an implemented oracle assignment or a proof against communication-aware partitioning.',
  'Current limitations matter: every GPU still reserves and scans a dense requested-ID address space; network-sized summaries/index arrays remain replicated; phase barriers prevent overlap across exchange boundaries. Three GPUs therefore do not receive one-third of all work. Some overhead is an implementation property that a better partition cannot eliminate.',
  'Nsight interpretation: long-scoreboard stalls identify dependencies on memory operations inside sampled kernels. Low achieved occupancy can reflect launch size, registers and active execution; it is not the fraction of wall time GPUs are busy. Traffic kernels read local owned/ghost buffers, so their long-scoreboard metric is not a direct measurement of NVLink waiting. Transfers and CPU synchronization outside kernels require stage timings, telemetry and ultimately a timeline trace.',
  'The hardware baseline demonstrated fast large peer copies, but these simulation exchanges are frequent and synchronization-sensitive. Effective bytes divided by an entire exchange stage includes packing, host count checks and waits; it is not raw NVLink bandwidth.' ])
 oracle=pd.read_csv(R/'restricted-oracle.csv')
 if not oracle.empty:
  ot=oracle.copy();ot['Calibration cost (s)']=ot.calibration_cost_ms/1000;ot['Evaluation (s)']=ot.evaluation_median_ms/1000;ot['Charged total (s)']=ot.charged_oracle_ms/1000
  for col in ['Calibration cost (s)','Evaluation (s)','Charged total (s)']:ot[col]=ot[col].map(lambda x:f'{x:.3f}')
  tablepage('Restricted measured-cost selector: independent evaluation',ot[['case','selected_policy','Evaluation (s)','Calibration cost (s)','Charged total (s)']].rename(columns={'case':'Case','selected_policy':'Chosen policy'}),'The excluded repeat-0 warm-up chooses among three policies; repetitions 1–5 evaluate that fixed choice. Calibration charges compute, setup and assignment for all three trials. This exploratory selector does not generate a new partition and is not a guaranteed optimal oracle. A supplied-choice result omits calibration; the charged result includes it.')
 textpage('4. Findings to take to the PI and next decision gate',[
  'Established: a physically validated three-GPU ownership protocol, concurrent peer transport, a shared policy interface, reproducible population-dependent workloads, repeated timing, allocation diagnostics and separate hardware-counter measurements.',
  'Not established: a learned prior, a positive multi-GPU speedup over the optimized one-GPU baseline, an oracle that constructs new partitions, or held-out/cross-network generalization. The restricted measured-cost selector only chooses among the three tested policies. Do not present the present policy differences as a published allocator result.',
  'Next: use these stage/communication measurements to decide whether sparse partition-local work and reduced synchronization are necessary baseline corrections. Keep physics and policy interfaces fixed while making any correction; then rerun this same matrix.',
  'Only after a viable baseline: measure regional task and boundary costs; construct an informed oracle partition on separate repetitions; charge construction/migration. If it offers meaningful net savings, predict those costs from topology, expected route flows, junction requests, and the persistence of boundary activity.',
  'Final research gate stays unchanged: a prior must beat the strongest inexpensive policy by 10% geometric-mean complete-runtime improvement on held-out cases, with confidence support, no >5% median case regression, exact correctness and the same memory budget. This development study does not satisfy or reject that future criterion.' ])
 textpage('Evidence, reproducibility and interpretation references',[
  'Primary evidence: timing-summary.csv and all-runs.csv; diagnostic-windows.csv and state-cost.csv; allocations.csv, memory.csv and populations.csv; nsight-raw.csv, nsight-summary.csv and nsight-state-matched.csv. Figures have standalone PNG and PDF versions. Raw .ncu-rep files remain in the profile archive.',
  'Reproduction: research/multigpu/a100_study/PROTOCOL.md defines the study and exploratory oracle analysis. Source/input freeze is artifacts/multigpu-preparation-20261027. Each cloud archive includes command logs, hashes, hardware records and results; later archives also embed study sources. The concurrent CUDA source hash matches the timed medium and large allocations.',
  'NVIDIA metric interpretation: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html . The Warp Stall Reasons section defines long scoreboard as waiting on a scoreboard dependency for L1TEX operations; this does not establish a bandwidth bottleneck or identify a particular source-level load.',
  'Profiler CLI and replay: https://docs.nvidia.com/nsight-compute/NsightComputeCli/ . We use separate kernel replay and --clock-control none. Actual clocks are recorded, not locked. Selected kernel counters cannot replace an unprofiled concurrency timeline.',
  'Modal rates and accounting: https://modal.com/pricing . Compute is bounded by application timeouts and a conservative local reservation guard. Billing JSON is posted usage, not a final invoice. No continuing GPU service is required to view this report.' ])
for _,fig in figures:plt.close(fig)
print(json.dumps(summary,indent=2))
