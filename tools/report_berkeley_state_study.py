#!/usr/bin/env python3
"""State-cost evidence and held-out, pre-run-prior prediction study.

Regional replay latency is a diagnostic target, not additive full-run latency.
No runtime allocation speedup is claimed from regression or a load proxy.
"""
import csv,json,math,heapq,statistics,hashlib,shutil,textwrap
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from berkeley_state_study import ROOT,ART,NET,dump,sha
OUT=ROOT/'reports/berkeley-state-study-20261014';OUT.mkdir(parents=True,exist_ok=True)
def csvwrite(path,rows):
 if not rows:return
 with path.open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def shortcase(name):
 p=name.split('_');return ('D' if p[0]=='distributed' else 'C')+p[1]+'/s'+p[2][-1]+'/'+('72' if p[3]=='mode0' else 'AP')
def fmt(value):return f'{value:.3f}' if math.isfinite(value) else 'Unreliable'
def phase(m,t,mode):
 if mode==2:return 2
 p=(t+(36 if m['axis']==1 else 0))%72
 return 2 if (24<=p<32 if m['turn']==1 else p<20) else 0
def prior(case,roads,moves,regions):
 """Deterministic point-queue forecast using only demand, topology and signals.

No receiving-space/spillback feedback, acceleration, future states or timings.
Intentionally evaluated as a fallible baseline, not a calibrated traffic model.
"""
 lines=(ART/'cases'/case['name']/'scenario.txt').read_text().splitlines()[2:]
 trips=[]
 for line in lines:
  x=line.split();n=int(x[1]);trips.append((float(x[0]),list(map(int,x[2:2+n])),list(map(int,x[2+n:2+2*n])),list(map(int,x[2+2*n:2+3*n-1]))))
 events=[];origin_next=defaultdict(float);lane_next=defaultdict(float);intervals=[];movement_intervals=[]
 for ident,(dep,edges,lanes,ms) in enumerate(trips):
  key=(edges[0],lanes[0]);admit=max(dep,origin_next[key]);origin_next[key]=admit+2.5
  intervals.append((dep,admit,regions[edges[0]],'pending',0))
  heapq.heappush(events,(admit,ident,0))
 while events:
  enter,ident,k=heapq.heappop(events);dep,edges,lanes,ms=trips[ident];r=roads[edges[k]];region=regions[edges[k]];arrival=enter+r['length']/r['speed']
  intervals.append((enter,arrival,region,'moving',r['speed']))
  intervals.append((max(enter,arrival-30/r['speed']),arrival,region,'approach',0))
  if k==len(edges)-1:continue
  m=moves[ms[k]];key=(edges[k],lanes[k]);release=max(arrival,lane_next[key])
  # Protected green only: conservative prior omits right-on-red and spillback.
  while phase(m,release,case['mode'])!=2:release+=.25
  lane_next[key]=release+2
  movement_intervals.append((enter,release,region,m['id']))
  intervals.append((arrival,release,region,'waiting',0));finish=release+m['length']/(7 if m['turn']==0 else 4)
  intervals.append((release,finish,region,'connector',0));heapq.heappush(events,(finish,ident,k+1))
 result={}
 for t in (60,180,450):
  for region in range(16):result[t,region]=dict(prior_active=0,prior_waiting=0,prior_approach=0,prior_connector=0,prior_pending=0,prior_green_active=0)
  for start,end,region,state,speed in intervals:
   if start<=t<end:
    r=result[t,region]
    if state in ('moving','waiting'):r['prior_active']+=1
    if state=='waiting':r['prior_waiting']+=1
    if state=='approach':r['prior_approach']+=1
    if state=='connector':r['prior_connector']+=1
    if state=='pending':r['prior_pending']+=1
  for start,end,region,mid in movement_intervals:
   if start<=t<end and phase(moves[mid],t,case['mode'])==2:result[t,region]['prior_green_active']+=1
 return result
def fit(train,features):
 X=np.array([[1.]+[float(r[f]) for f in features] for r in train]);y=np.array([r['motion_ms'] for r in train]);scale=np.maximum(1.,np.std(X,axis=0));Z=X/scale
 # Fixed weak ridge; no hyperparameters selected using the held-out seed.
 penalty=np.eye(Z.shape[1])*.01;penalty[0,0]=0
 beta=np.linalg.solve(Z.T@Z+penalty,Z.T@y)/scale
 return lambda r:max(0.,float(np.dot([1.]+[float(r[f]) for f in features],beta))),beta.tolist()
def main():
 cfg=json.loads((ART/'prepared.json').read_text());geo=json.loads((NET/'geometry.json').read_text());roads,moves=geo['roads'],geo['movements'];regions=list(map(int,(NET/'regions.txt').read_text().split()))
 runrows=[];windows=[];regional=[];state_replays=[];speed_replays=[];scanrows=[];gpu=[];state_gpu=[];source_samples=[];controls=[]
 for case in cfg['cases']:
  name=case['name'];runs=[json.loads((ART/'runs'/name/str(k)/'validated.json').read_text()) for k in (1,2,3)]
  m=runs[0]['metrics'];series=list(csv.DictReader((ART/'runs'/name/'0'/'step_metrics.csv').open()))
  active_samples=[float(x['road_stopped'])+float(x['road_moving']) for x in series];stopped_samples=[float(x['road_stopped']) for x in series]
  fraction=sum(stopped_samples)/max(1,sum(active_samples))
  runrows.append(dict(case=name,n=case['n'],mode=case['mode'],seed=case['seed'],completed=m['completed'],pending=m['pending'],active=m['active'],stopped_vehicle_time_fraction=fraction,median_gpu_ms=statistics.median(r['metrics']['advance_gpu_ms']+r['metrics']['junction_gpu_ms'] for r in runs),median_index_plus_motion_ms=statistics.median(r['metrics']['advance_gpu_ms'] for r in runs),median_junction_ms=statistics.median(r['metrics']['junction_gpu_ms'] for r in runs),median_wall_s=statistics.median(r['wall_seconds'] for r in runs),gap_failures=m['gap_failures'],conflict_failures=m['conflict_failures']))
  forecast=prior(case,roads,moves,regions)
  for t in (60,180,450):
   pop=next(x for x in series if float(x['time'])==t);windows.append(dict(case=name,**pop))
   folder=ART/'replays'/name/str(t)
   normal=json.loads((folder/'selector_-1.json').read_text());instrument=json.loads((folder/'instrumented.json').read_text());assert normal['physical_digest']==instrument['physical_digest']
   for selector in range(22):
    r=json.loads((folder/f'selector_{selector}.json').read_text());assert r['physical_match'];r.update(case=name,seed=case['seed'],mode=case['mode'],requested=case['n'],motion_ms=statistics.median(r['motion_ms']),total_ms=statistics.median(r['total_ms']),mean_speed=r['speed_sum']/max(1,r['active']))
    r.pop('physical_digest',None);r.pop('physical_match',None)
    if selector<16:r.update(forecast[t,selector]);regional.append(r)
    else:state_replays.append(r)
   for speed in (0,5):
    r=json.loads((folder/f'speed_{speed}.json').read_text());speed_replays.append(dict(case=name,time=t,input_road_speed=speed,active=r['active'],motion_ms=statistics.median(r['motion_ms']),total_ms=statistics.median(r['total_ms'])))
   for file in folder.glob('control_*.json'):
    r=json.loads(file.read_text());label=file.stem.removeprefix('control_')
    if label not in ('half_active','mix_all','mix_half'):assert r['physical_digest']==normal['physical_digest']
    baseline=json.loads((folder/'control_mix_all.json').read_text()) if label=='mix_half' else normal
    controls.append(dict(case=name,time=t,control=label,active=r['active'],stopped=r['stopped'],baseline_ms=statistics.median(baseline['total_ms']),variant_ms=statistics.median(r['total_ms']),ratio=statistics.median(r['total_ms'])/statistics.median(baseline['total_ms'])))
   observations=list(csv.DictReader((folder/'instrumented.json.counts.csv').open()))
   buckets=defaultdict(list)
   for r in observations:
    state=int(r['state']);label='pending' if state==0 else 'connector' if state==2 else 'completed' if state==3 else ('stopped' if float(r['speed'])<.5 else 'moving')+('_approach' if float(r['distance_to_stop'])<30 else '_interior')
    if state==1:label+=('_leader' if int(r['leader'])<0 else '_follower')
    buckets[label].append(r)
   for label,rs in buckets.items():scanrows.append(dict(case=name,time=t,state=label,count=len(rs),receiving_candidates=sum(int(r['receiving_candidates']) for r in rs),connector_candidates=sum(int(r['connector_candidates']) for r in rs),median_thread_cycles=statistics.median(int(r['thread_cycles']) for r in rs)))
   p=ART/'profiles'/name/str(t)/'metrics.csv'
   if p.exists():
    raw=list(csv.DictReader(p.open()));units=raw[0]
    for row in raw[1:]:
     def val(k):return float(row[k].replace(',',''))
     duration=val('gpu__time_duration.sum')*{'ns':1e-6,'us':1e-3,'µs':1e-3,'ms':1,'s':1000}[units['gpu__time_duration.sum']]
     gpu.append(dict(case=name,time=t,kernel=row['Kernel Name'].split('(')[0],duration_ms=duration,occupancy=val('sm__warps_active.avg.pct_of_peak_sustained_active'),eligible=val('smsp__warps_eligible.avg.per_cycle_active'),threads=val('smsp__thread_inst_executed_per_inst_executed.ratio'),issue=val('smsp__issue_active.avg.pct_of_peak_sustained_active'),long_scoreboard=val('smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),sm_throughput=val('sm__throughput.avg.pct_of_peak_sustained_elapsed'),l1_throughput=val('l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),l2_throughput=val('lts__throughput.avg.pct_of_peak_sustained_elapsed'),instructions=val('smsp__inst_executed.sum'),global_load_sectors=val('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),local_load_sectors=val('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),registers=val('launch__registers_per_thread'),grid=row['Grid Size'],block=row['Block Size']))
   for p in (ART/'profiles'/name/str(t)).glob('state_*/metrics.csv'):
    raw=list(csv.DictReader(p.open()));row=raw[-1];selector=int(p.parent.name.split('_')[1]);counts=json.loads((folder/f'selector_{selector}.json').read_text())
    state_gpu.append(dict(case=name,time=t,selector=selector,active=counts['active'],stopped=counts['stopped'],connector=counts['connector'],pending=counts['pending'],occupancy=float(row['sm__warps_active.avg.pct_of_peak_sustained_active']),eligible=float(row['smsp__warps_eligible.avg.per_cycle_active']),threads=float(row['smsp__thread_inst_executed_per_inst_executed.ratio']),long_scoreboard=float(row['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'])))
   source_file=ART/'profiles'/name/str(t)/'source.csv'
   if source_file.exists():
    raw=list(csv.reader(source_file.open()));h=next(i for i,r in enumerate(raw) if 'stall_long_sb' in r);column=raw[h].index('stall_long_sb');samples=[dict(case=name,time=t,line=int(r[0]),code=r[1],samples=float(r[column].replace(',',''))) for r in raw[h+1:] if len(r)>column and r[0].isdigit() and r[2]=='-'];total=sum(r['samples'] for r in samples)
    for r in samples:r['share_percent']=100*r['samples']/max(1,total)
    source_samples.extend(sorted(samples,key=lambda r:-r['samples']))
 quality=[];retries=[]
 for p in (ART/'profiles').glob('*/*/occupancy_retry/metrics.csv'):
  for row in list(csv.DictReader(p.open()))[1:]:retries.append(dict(case=p.parents[2].name,time=int(p.parents[1].name),kernel=row['Kernel Name'].split('(')[0],occupancy=float(row['sm__warps_active.avg.pct_of_peak_sustained_active']),protocol='Single requested metric; cache-control all; separate sample'))
 for r in gpu+state_gpu:
  flags=[]
  for key in ('occupancy','issue','long_scoreboard','sm_throughput','l1_throughput','l2_throughput'):
   if key in r and not (0<=r[key]<=100):
    quality.append(dict(case=r['case'],time=r['time'],kernel=r.get('kernel',str(r.get('selector'))),metric=key,raw_value=r[key]));r[key]=float('nan');flags.append(key)
  r['quality_flags']=','.join(flags)
 dump(OUT/'counter_quality.json',dict(invalid_original_metrics=quality,occupancy_retries=retries,interpretation='Out-of-range metrics are not clamped or interpreted as physical occupancy. Narrow-metric, cold-cache retries use a different protocol. Exact cause of original anomaly is unproven.',reference='https://docs.nvidia.com/nsight-compute/ProfilingGuide/#range-and-precision'))
 csvwrite(OUT/'occupancy_retries.csv',retries)
 for name,rows in [('runs',runrows),('windows',windows),('regional_replays',regional),('state_replays',state_replays),('speed_replays',speed_replays),('scan_counts',scanrows),('gpu_metrics',gpu)]:csvwrite(OUT/f'{name}.csv',rows)
 csvwrite(OUT/'state_gpu_metrics.csv',state_gpu);csvwrite(OUT/'source_samples.csv',source_samples)
 csvwrite(OUT/'controlled_replays.csv',controls)
 models={'requested_only':['requested'],'observed_active':['requested','active'],'observed_speed':['requested','mean_speed'],'observed_states':['requested','active','stopped','approach','requests','connector','conflict_degree_sum'],'prior_active':['requested','prior_active'],'prior_states':['requested','prior_active','prior_waiting','prior_approach','prior_connector','prior_pending','prior_green_active']}
 predictions=[];scores=[];coefficients={}
 for split in ('held_out_seed','held_out_seed_and_regions'):
  train=[r for r in regional if r['seed']==1401 and (split=='held_out_seed' or r['selector']<12)]
  test=[r for r in regional if r['seed']==1402 and (split=='held_out_seed' or r['selector']>=12)]
  y=np.array([r['motion_ms'] for r in test])
  for name,features in models.items():
   predict,beta=fit(train,features);pred=np.array([predict(r) for r in test]);mae=float(np.mean(abs(y-pred)));r2=float(1-np.sum((pred-y)**2)/max(1e-20,np.sum((y-y.mean())**2)))
   scores.append(dict(split=split,model=name,train_windows=len(train),test_windows=len(test),mae_ms=mae,r2=r2));coefficients[split+'/'+name]=dict(features=['intercept']+features,coefficients=beta)
   for r,p in zip(test,pred):predictions.append(dict(split=split,model=name,case=r['case'],time=r['time'],region=r['selector'],observed_ms=r['motion_ms'],predicted_ms=float(p)))
 csvwrite(OUT/'prediction_scores.csv',scores);csvwrite(OUT/'predictions.csv',predictions);dump(OUT/'prediction_models.json',coefficients)
 # Pre-run fixed assignments: a diagnostic load proxy, NOT measured GPU speedup.
 predict,_=fit([r for r in regional if r['seed']==1401],models['prior_states']);partition_results=[]
 for case in cfg['cases']:
  if case['seed']!=1402:continue
  rows=[r for r in regional if r['case']==case['name']];weights={region:sum(predict(r) for r in rows if r['selector']==region) for region in range(16)};actual={region:sum(r['motion_ms'] for r in rows if r['selector']==region) for region in range(16)}
  prior_counts={region:sum(r['prior_active'] for r in rows if r['selector']==region) for region in range(16)}
  assignments={}
  for policy,w in [('prior_cost',weights),('prior_vehicle_count',prior_counts)]:
   loads=[0.]*4;assignment={}
   for region in sorted(w,key=lambda k:(-w[k],k)):
    target=min(range(4),key=lambda k:(loads[k],k));assignment[region]=target;loads[target]+=w[region]
   assignments[policy]=assignment
  assignments['equal_geography']={r:r//4 for r in range(16)}
  for policy,assignment in assignments.items():
   loads=[sum(actual[r] for r in range(16) if assignment[r]==k) for k in range(4)]
   partition_results.append(dict(case=case['name'],policy=policy,max_diagnostic_load_ms=max(loads),max_over_mean=max(loads)/max(1e-9,statistics.mean(loads)),assignment=json.dumps(assignment)))
 csvwrite(OUT/'static_partition_proxy.csv',partition_results)
 score={(r['split'],r['model']):r for r in scores}
 seed_base=score['held_out_seed','prior_active'];seed_state=score['held_out_seed','prior_states'];geo_state=score['held_out_seed_and_regions','prior_states']
 speed_lookup={(r['case'],r['time'],r['input_road_speed']):r['total_ms'] for r in speed_replays}
 speed_ratio=statistics.median(speed_lookup[c,t,5]/speed_lookup[c,t,0] for c,t,s in speed_lookup if s==0)
 ratio={label:statistics.median(r['ratio'] for r in controls if r['control']==label) for label in set(r['control'] for r in controls)}
 findings=[
  'Full-network execution is enabled and validated for the recorded research cases: 48 full runs (12 pilot/trajectory runs plus 36 timing runs), 36 saved-state windows, 11 four-way small-network equivalence cases. Root Compute Sanitizer reported zero errors for one citywide restart. This is not field calibration or exhaustive memory verification.',
  f'Changing every road vehicle from 0 to 5 m/s at fixed positions/populations changes median total replay time by a factor of {speed_ratio:.4f}. There is no material speed-only effect in these tests; no significance claim is made for the small difference.',
  f'Doubling total records with inactive padding costs {ratio["padding"]:.3f}x. Halving road-active population at an exactly 50/50 stopped/moving input mix costs {ratio["mix_half"]:.3f}x. Index rebuilds, record copies and remaining scans prevent simple proportional scaling.',
  f'Shuffling records costs {ratio["shuffle"]:.3f}x; lane ordering {ratio["lane_order"]:.3f}x; 64-thread blocks {ratio["block64"]:.3f}x; 256-thread blocks {ratio["block256"]:.3f}x. These are medians across snapshots, not universal recommendations.',
  'At 180 s with signals, distributed demand has 1,975 road vehicles and 11 pending; concentrated demand has 307 road vehicles and 1,739 pending. Motion takes 3.440 versus 1.790 ms in separate Nsight samples; occupancy is 16.87% versus 9.05%. Requested count and a congestion label do not specify sustained GPU work.',
  'Source samples locate roughly 97% of mapped long-scoreboard samples in the remaining receiving-space and connector scans for both profiled cases. Every road-active vehicle checks all 2,048 records for connectors, including stopped followers. Receiving searches vary with signal eligibility, blockers and their storage position. Source shares are not elapsed-time fractions.',
  'Inactive records can have long instrumented thread-clock intervals despite zero scans when sharing execution with active work. Clock deltas include warp reconvergence and scheduling effects; they cannot be interpreted as each vehicle\'s isolated compute demand. Connector/pending filtered motion kernels primarily copy records; their high stall percentages do not measure connector propagation or admission cost.',
  f'On the held-out seed, the pre-run state prior reduces regional replay MAE only {(1-seed_state["mae_ms"]/seed_base["mae_ms"])*100:.2f}% versus the pre-run active-count prior ({seed_base["mae_ms"]:.3f} to {seed_state["mae_ms"]:.3f} ms). R² is {seed_state["r2"]:.3f}. With geography also held out, R² is {geo_state["r2"]:.3f}: this prior does not establish geographic generalization.',
  'A four-queue static partition built from pre-run predicted costs is supplied as an offline diagnostic load proxy. Regional filtered-kernel times are not additive full-kernel costs, so this table is not a measured allocator speedup. No static GPU scheduler was deployed.',
  'Conclusion: admission/population and executed search work are stronger leads than speed alone, but the present evidence does not justify claiming a transferable static allocation algorithm. Remaining global scans and limited queue/spillback prediction must be addressed and the held-out test repeated before making that claim.'
 ]
 (OUT/'conclusions.md').write_text('# Conclusions\n\n'+'\n\n'.join(findings)+'\n\nCounter-quality note: out-of-range occupancy readings are retained but excluded from physical interpretations; narrower, cold-cache retries are separate samples. See [NVIDIA range and precision guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#range-and-precision).\n')
 dump(OUT/'summary.json',dict(network=cfg['network'],runs=runrows,prediction_scores=scores,validation=json.loads((ART/'validation.json').read_text()),scope='Full-network synthetic research trajectories plus immutable motion replays. Regional timing and partition sums are diagnostic proxies, not measured allocation speedups. Observed-state models use actual states; only prior models are available pre-run.',timing_repeats=[1,2,3],pilot_repeat=0,prior='Deterministic point queues; 2.5 s origin admission headway, 2 s intersection service headway; free-flow road travel; known green schedules; no spillback, right-on-red, acceleration or future trajectory input.'))
 with PdfPages(OUT/'berkeley-state-study.pdf') as pdf:
  def table(title,columns,rows,filename):
   fig,ax=plt.subplots(figsize=(16,9));ax.axis('off');ax.set_title(title,pad=20);tab=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.8)
   for (r,c),cell in tab.get_celld().items():
    if r==0:cell.set_facecolor('#dcebee')
   fig.text(.03,.025,'D: distributed; C: concentrated; s1/s2: seeds 1401/1402; 72: signals; AP: always-permitted. See CSV files for full case IDs.',fontsize=9);fig.tight_layout(rect=(0,.05,1,1));pdf.savefig(fig);fig.savefig(OUT/filename,dpi=140);plt.close(fig)
  for page in range(2):
   fig=plt.figure(figsize=(11.7,16.5));content='Berkeley state-cost study: findings\n\n'+'\n\n'.join(textwrap.fill(x,94) for x in findings[page*5:page*5+5]);fig.text(.06,.95,content,va='top',fontsize=12,linespacing=1.5);pdf.savefig(fig);plt.close(fig)
  table('Berkeley research model: 600 s, median of three timing repeats',['Case','Completed','Pending','Stopped vehicle-time','GPU ms','Index + motion ms','Junction ms'],[[shortcase(r['case']),r['completed'],r['pending'],f"{100*r['stopped_vehicle_time_fraction']:.1f}%",f"{r['median_gpu_ms']:.1f}",f"{r['median_index_plus_motion_ms']:.1f}",f"{r['median_junction_ms']:.1f}"] for r in runrows],'runs.png')
  for t in (60,180,450):
   selected=[r for r in gpu if r['time']==t and 'advance' in r['kernel']]
   table(f'Motion kernel, t={t} s: matched snapshots; index build excluded',['Case','ms','Occ %','Eligible','Threads/inst','Issue %','Long scoreboard %','L1 %','L2 %'],[[shortcase(r['case'])]+[fmt(r[k]) for k in ('duration_ms','occupancy','eligible','threads','issue','long_scoreboard','l1_throughput','l2_throughput')] for r in selected],f'gpu_{t}.png')
  for label in ('distributed_2048','concentrated_2048'):
   name=label+'_seed1401_mode0';selected=[r for r in gpu if r['case']==name and r['time']==180]
   table(label+' at 180 s: separate kernel counters',['Kernel','ms','Occ %','Eligible','Threads/inst','Issue %','Long scoreboard %'],[[r['kernel'].replace('<unnamed>::','')]+[fmt(r[k]) for k in ('duration_ms','occupancy','eligible','threads','issue','long_scoreboard')] for r in selected],label+'_kernels.png')
  labels={16:'Moving road interior',17:'Stopped road',18:'Within 30 m of stop line',19:'Connector',20:'Pending'}
  table('State-filtered diagnostic motion replays: filtering changes launch behavior',['Case','Selected work','Road active','Occ %','Eligible','Threads/inst','Long scoreboard %'],[[shortcase(r['case']),labels[r['selector']],r['active']]+[fmt(r[k]) for k in ('occupancy','eligible','threads','long_scoreboard')] for r in state_gpu],'state_gpu.png')
  for label in ('distributed_2048','concentrated_2048'):
   selected=[r for r in source_samples if r['case']==label+'_seed1401_mode0'][:6]
   table(label+': source-correlated long-scoreboard samples (not time fractions)',['CUDA line','Source context','Samples','Share %'],[[r['line'],textwrap.fill(r['code'][:110],55),int(r['samples']),f"{r['share_percent']:.1f}"] for r in selected],label+'_source.png')
  table('Held-out prediction of regional replay latency (not full-run latency)',['Holdout','Model','MAE ms','R²'],[[r['split'],r['model'],f"{r['mae_ms']:.5f}",f"{r['r2']:.3f}"] for r in scores],'predictions.png')
  table('Controlled immutable replays: total time includes index construction',['Control','Median variant / original','Min','Max'],[[label]+[f'{fn([r["ratio"] for r in controls if r["control"]==label]):.3f}' for fn in (statistics.median,min,max)] for label in sorted(set(r['control'] for r in controls))],'controls.png')
  notes='''Scope and interpretation

Network: 5,013 roads, 12,216 inferred movements, 1,672 junction groups.
Geometry uses endpoint chords; lane turns and signal plans are synthetic.
These are validated model trajectories, not field-calibrated Berkeley traffic.

All 11 small-network scenarios match the frozen reference, scalable serial,
scalable parallel and indexed parallel implementations exactly. Full runs
must pass all implemented movement/gap checks and repeat deterministically.
Three measured repeats follow a separate trajectory/pilot run. Fixed clocks.

Motion indexing is rebuilt each step; full-run index + motion timing includes
that cost. Six-kernel hardware tables exclude index construction kernels;
complete indexed pipeline timing is supplied separately. Profiling is separate.
GPU totals exclude CPU auditing, trajectory output and host readback; wall time
includes them. Do not interpret instrumented thread-cycle sums as GPU elapsed time.

Regional replay retains all vehicles as read context and filters only output
work. Filtering changes occupancy and cache behavior. Regional latencies and
the static-partition load proxy are not additive full-run costs or speedups.
Speed interventions hold positions/populations fixed but are constructed
one-step performance tests, not calibrated trajectories. Motion storage, padding
and block-size variants preserve canonical results; admission-ID shuffles are
not tested. Population removal can change gaps and slightly change the speed mix.

Observed-state predictors are diagnostic oracles. Prior predictors see only
routes, departures, topology and controllers. Seed 1402 is held out; a separate
split also holds out four geographic regions. Point queues omit spillback and
right-on-red, so concentrated demand can expose forecast errors.

No deployed static GPU allocator or measured allocator speedup is claimed.
Ordinary CUDA launches do not reserve SMs by traffic region. A future work-queue
policy must beat vehicle-count and geographic baselines in end-to-end runtime,
including prediction, construction and dispatch, with identical physics.
'''
  fig=plt.figure(figsize=(11.7,16.5));fig.text(.06,.95,notes,va='top',fontsize=12,linespacing=1.5);pdf.savefig(fig);plt.close(fig)
 import html
 (OUT/'README.md').write_text('# Berkeley state-cost study\n\nSee `berkeley-state-study.pdf`, `conclusions.md`, `summary.json`, and CSV tables. Full-run results use repeats 1–3; repeat 0 supplies trajectories. Static partition scores are diagnostic replay-load proxies, not measured GPU speedups.\n')
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Berkeley state-cost study</title><style>body{font:17px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.5}img{width:100%;border:1px solid #ddd}li{margin-bottom:16px}</style><h1>Berkeley state-cost study</h1><p><a href="berkeley-state-study.pdf">PDF report</a> · <a href="gpu_metrics.csv">GPU counters</a> · <a href="prediction_scores.csv">Held-out prediction scores</a> · <a href="conclusions.md">Conclusions</a></p><ol>'+''.join('<li>'+html.escape(x)+'</li>' for x in findings)+'</ol>'+''.join('<h2>'+html.escape(p.stem.replace('_',' '))+'</h2><img src="'+p.name+'">' for p in sorted(OUT.glob('*.png'))))
 print(OUT/'berkeley-state-study.pdf')
if __name__=='__main__':main()
