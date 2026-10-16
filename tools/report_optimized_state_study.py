#!/usr/bin/env python3
"""Matched optimized baseline, controlled factors, and held-out state-cost tests."""
import json,csv,statistics,math,html
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import report_berkeley_state_study as old
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-optimized-state-20261016';B=R/'artifacts/current/berkeley-state-study-20261014';O=R/'reports/berkeley-optimized-state-20261016';N=R/'data/networks/berkeley_junction_study'
old.ART=A
def load(p):return json.loads(p.read_text())
def write(name,rows):
 if not rows:return
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stage(name):
 for text,label in [('initializeLaneOrder','Leader initialization'),('linkLaneLeaders','Leader linking'),('collectMotionContext','Context collection'),('resolveMotionAvailability','Movement availability'),('advance','Road motion'),('prepareJunctions','Junction preparation'),('summarizeJunctions','Junction summaries'),('requestJunctionEntry','Junction requests'),('arbitrateJunctions','Junction arbitration'),('spawnJunctionVehicles','Spawn')]:
  if text in name:return label
 return 'Leader sorting'
def gpurows(path,case,t,selector=-1):
 raw=list(csv.DictReader(path.open()));units=raw[0];out=[]
 mapping={'duration_ms':'gpu__time_duration.sum','occupancy':'sm__warps_active.avg.pct_of_peak_sustained_active','eligible':'smsp__warps_eligible.avg.per_cycle_active','threads':'smsp__thread_inst_executed_per_inst_executed.ratio','issue':'smsp__issue_active.avg.pct_of_peak_sustained_active','long_scoreboard':'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','sm_throughput':'sm__throughput.avg.pct_of_peak_sustained_elapsed','l1_throughput':'l1tex__throughput.avg.pct_of_peak_sustained_elapsed','l2_throughput':'lts__throughput.avg.pct_of_peak_sustained_elapsed','instructions':'smsp__inst_executed.sum','global_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','local_load_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','registers':'launch__registers_per_thread'}
 for x in raw[1:]:
  r=dict(case=case,time=t,selector=selector,kernel=x['Kernel Name'],stage=stage(x['Kernel Name']))
  for label,key in mapping.items():
   value=x.get(key,'').replace(',','');r[label]=float(value) if value not in ('','n/a','N/A') else float('nan')
  r['duration_ms']*= {'ns':1e-6,'us':1e-3,'µs':1e-3,'ms':1,'s':1000}[units['gpu__time_duration.sum']]
  r['quality_flags']=','.join(k for k in ['occupancy','issue','long_scoreboard','sm_throughput','l1_throughput','l2_throughput'] if not 0<=r[k]<=100)
  out.append(r)
 return out
def pct(x):return f'{x:.2f}' if 0<=x<=100 else 'Unreliable'
def main():
 O.mkdir(exist_ok=True);cfg=load(A/'prepared.json');geo=load(N/'geometry.json');roads,moves=geo['roads'],geo['movements'];regions=np.array(list(map(int,(N/'regions.txt').read_text().split())))
 dtype=np.dtype({'names':['state','k','n','mid','edges','lanes','moves','pos','v'],'formats':['i4','i4','i4','i4',('i4',32),('i4',32),('i4',32),'f4','f4'],'offsets':[0,4,8,12,24,152,280,412,416],'itemsize':428})
 with (N/'model.txt').open() as model:
  nr,nm,nj=map(int,model.readline().split())
  for _ in range(nr+nm):model.readline()
  fmt,count=model.readline().split();degree=np.zeros(nm,dtype=int)
  for _ in range(int(count)):a,b=map(int,model.readline().split());degree[a]+=1
 top={i:dict(top_roads=int((regions==i).sum()),top_lanes=sum(r['lanes'] for r in roads if regions[r['id']]==i),top_movements=sum(regions[m['incoming']]==i for m in moves)) for i in range(16)}
 rows=[];states=[];windows=[];runs=[];controls=[];gpu=[];sgpu=[];scans=[];whole=[]
 for c in cfg['cases']:
  name=c['name'];fore=old.prior(c,roads,moves,regions)
  optimized=[load(A/'runs'/name/str(k)/'validated.json') for k in (1,2,3)];baseline=[load(B/'runs'/name/str(k)/'validated.json') for k in (1,2,3)]
  m=optimized[0]['metrics'];runs.append(dict(case=name,completed=m['completed'],pending=m['pending'],active=m['active'],optimized_gpu_ms=statistics.median(x['metrics']['advance_gpu_ms']+x['metrics']['junction_gpu_ms'] for x in optimized),old_gpu_ms=statistics.median(x['metrics']['advance_gpu_ms']+x['metrics']['junction_gpu_ms'] for x in baseline),optimized_index_motion_ms=statistics.median(x['metrics']['advance_gpu_ms'] for x in optimized),optimized_junction_ms=statistics.median(x['metrics']['junction_gpu_ms'] for x in optimized),wall_s=statistics.median(x['wall_seconds'] for x in optimized),gpu_ms_min=min(x['metrics']['advance_gpu_ms']+x['metrics']['junction_gpu_ms'] for x in optimized),gpu_ms_max=max(x['metrics']['advance_gpu_ms']+x['metrics']['junction_gpu_ms'] for x in optimized)))
  series=list(csv.DictReader((A/'runs'/name/'0/step_metrics.csv').open()))
  for t in (60,180,450):
   d=A/'replays'/name/str(t);normal=load(d/'selector_-1.json');window=dict(case=name,**next(r for r in series if float(r['time'])==t));cars=np.frombuffer((A/'runs'/name/'0'/f'snapshot_{t}.bin').read_bytes()[8:],dtype=dtype);ix=np.arange(len(cars));edge=cars['edges'][ix,cars['k']];lane=cars['lanes'][ix,cars['k']];road=cars['state']==1;conn=cars['state']==2
   occupied={(int(edge[i]),int(lane[i])) for i in ix[road&(cars['pos']<14.5)]}
   obs=list(csv.DictReader((d/'instrumented.json.counts.csv').open()));leaders=np.array([int(x['leader']) for x in obs]);dist=np.array([r['length'] for r in roads])[edge]-cars['pos'];near=road&(dist<30);blocked=np.zeros(len(cars),dtype=bool)
   for i in ix[road&(cars['k']+1<cars['n'])]:
    movement=moves[int(cars['moves'][i,cars['k'][i]])];blocked[i]=(movement['outgoing'],movement['out_lane']) in occupied
   window.update(approaching_leaders=int((near&(leaders<0)).sum()),queued_followers=int((road&(cars['v']<.5)&(leaders>=0)).sum()),near_downstream_occupied=int((near&blocked).sum()),requests=normal['requests'],speed_p10=float(np.quantile(cars['v'][road],.1)) if road.any() else 0,speed_p50=float(np.quantile(cars['v'][road],.5)) if road.any() else 0,speed_p90=float(np.quantile(cars['v'][road],.9)) if road.any() else 0)
   windows.append(window)
   query_all={int(cars['moves'][i,cars['k'][i]]) for i in ix[road&(cars['k']+1<cars['n'])]}
   matched=[next(r for r in csv.DictReader((A/'runs'/name/str(rep)/'step_metrics.csv').open()) if float(r['time'])==t) for rep in (1,2,3)]
   whole.append(dict(case=name,time=t,seed=c['seed'],requested=c['n'],active=int(road.sum()),stopped=int((road&(cars['v']<.5)).sum()),connector=int(conn.sum()),pending=int((cars['state']==0).sum()),requests=normal['requests'],queried_movements=len(query_all),occupied_lanes=len({(int(edge[i]),int(lane[i])) for i in ix[road]}),conflict_edges=sum(int(degree[m]) for m in query_all),index_motion_ms=statistics.median(float(r['advance_gpu_ms']) for r in matched),junction_ms=statistics.median(float(r['junction_gpu_ms']) for r in matched),full_step_ms=statistics.median(float(r['advance_gpu_ms'])+float(r['junction_gpu_ms']) for r in matched)))
   for selector in range(22):
    r=load(d/f'selector_{selector}.json');r.update(case=name,seed=c['seed'],mode=c['mode'],requested=c['n'],motion_ms=statistics.median(r['motion_ms']),total_ms=statistics.median(r['total_ms']),mean_speed=r['speed_sum']/max(1,r['active']));r.pop('physical_digest');r.pop('physical_match')
    if selector<16:
     select=regions[edge]==selector;rr=road&select;query={int(cars['moves'][i,cars['k'][i]]) for i in ix[rr&(cars['k']+1<cars['n'])]};r.update(fore[t,selector]);r.update(top[selector]);r.update(occupied_lanes=len({(int(edge[i]),int(lane[i])) for i in ix[rr]}),queried_movements=len(query),queried_conflict_edges=sum(int(degree[m]) for m in query),connector_candidates=sum(int(obs[i]['connector_candidates']) for i in ix[rr]),approaching_leaders=int((near&select&(leaders<0)).sum()),queued_followers=int((rr&(cars['v']<.5)&(leaders>=0)).sum()),near_downstream_occupied=int((near&select&blocked).sum()));rows.append(r)
    else:states.append(r)
   for p in sorted(d.glob('control_*.json')):
    label=p.stem[8:];x=load(p);ref=load(d/'control_mix_all.json') if label=='mix_half' else normal;oldref=load(B/'replays'/name/str(t)/('control_mix_all.json' if label=='mix_half' else 'selector_-1.json'));oldvar=load(B/'replays'/name/str(t)/p.name)
    controls.append(dict(case=name,time=t,control=label,optimized_ratio=statistics.median(x['total_ms'])/statistics.median(ref['total_ms']),old_ratio=statistics.median(oldvar['total_ms'])/statistics.median(oldref['total_ms'])))
   x,y=load(d/'speed_0.json'),load(d/'speed_5.json');ox,oy=load(B/'replays'/name/str(t)/'speed_0.json'),load(B/'replays'/name/str(t)/'speed_5.json');controls.append(dict(case=name,time=t,control='speed5_vs0',optimized_ratio=statistics.median(y['total_ms'])/statistics.median(x['total_ms']),old_ratio=statistics.median(oy['total_ms'])/statistics.median(ox['total_ms'])))
   scans.append(dict(case=name,time=t,receiving_candidates=sum(int(x['receiving_candidates']) for x in obs),connector_candidates=sum(int(x['connector_candidates']) for x in obs),road_active=int(road.sum())))
   p=A/'profiles'/name/str(t)/'metrics.csv'
   if p.exists():gpu+=gpurows(p,name,t)
   for p in (A/'profiles'/name/str(t)).glob('state_*/metrics.csv'):sgpu+=gpurows(p,name,t,int(p.parent.name[6:]))
 models={'requested_only':['requested'],'active_count':['requested','active'],'topology':['requested','top_roads','top_lanes','top_movements'],'active_topology':['requested','active','top_roads','top_lanes','top_movements'],'observed_states':['requested','active','stopped','approach','requests','connector','conflict_degree_sum'],'observed_work':['requested','active','occupied_lanes','queried_movements','queried_conflict_edges','connector_candidates','connector','requests','approaching_leaders','queued_followers','near_downstream_occupied','top_movements'],'prior_active':['requested','prior_active'],'prior_states':['requested','prior_active','prior_waiting','prior_approach','prior_connector','prior_pending','prior_green_active']}
 scores=[];preds=[];coeff={}
 for split in ('held_out_seed','held_out_seed_and_regions'):
  train=[r for r in rows if r['seed']==1401 and (split=='held_out_seed' or r['selector']<12)];test=[r for r in rows if r['seed']==1402 and (split=='held_out_seed' or r['selector']>=12)];y=np.array([r['motion_ms'] for r in test])
  for name,features in models.items():
   predict,beta=old.fit(train,features);p=np.array([predict(r) for r in test]);scores.append(dict(split=split,model=name,train=len(train),test=len(test),mae_ms=float(np.mean(abs(y-p))),r2=float(1-np.sum((p-y)**2)/max(1e-20,np.sum((y-y.mean())**2)))));coeff[split+'/'+name]=dict(features=features,coefficients=beta)
   for r,pred in zip(test,p):preds.append(dict(case=r['case'],time=r['time'],region=r['selector'],model=name,split=split,observed_ms=r['motion_ms'],predicted_ms=float(pred)))
 for name,data in [('runs',runs),('windows',windows),('regional_replays',rows),('state_replays',states),('controls',controls),('scan_counts',scans),('gpu_metrics',gpu),('state_gpu_metrics',sgpu),('prediction_scores',scores),('predictions',preds)]:write(name,data)
 whole_scores=[]
 for target in ['full_step_ms','index_motion_ms','junction_ms']:
  dataset=[dict(r,motion_ms=r[target]) for r in whole];train=[r for r in dataset if r['seed']==1401];test=[r for r in dataset if r['seed']==1402];y=np.array([r['motion_ms'] for r in test])
  for name,features in {'requested_only':['requested'],'active_count':['requested','active'],'state_work':['requested','active','requests','connector','queried_movements','conflict_edges']}.items():
   predict,beta=old.fit(train,features);p=np.array([predict(r) for r in test]);whole_scores.append(dict(target=target,model=name,train=len(train),test=len(test),mae_ms=float(np.mean(abs(y-p))),r2=float(1-np.sum((p-y)**2)/max(1e-20,np.sum((y-y.mean())**2)))))
 write('whole_step_windows',whole);write('whole_step_scores',whole_scores)
 (O/'models.json').write_text(json.dumps(coeff,indent=2))
 agg=[]
 for control in sorted(set(r['control'] for r in controls)):
  v=[r for r in controls if r['control']==control];agg.append(dict(control=control,old_ratio=statistics.median(r['old_ratio'] for r in v),optimized_ratio=statistics.median(r['optimized_ratio'] for r in v)))
 extras=[]
 for p in (A/'extra_controls').glob('*/*/*/3.json'):
  x=load(p);label=p.parent.name;case=p.parents[2].name;t=int(p.parents[1].name);baseline=load(p.with_name('0.json'));normal=load(A/'replays'/case/str(t)/'selector_-1.json')
  extras.append(dict(case=case,time=t,control=label,active=x['active'],stopped=x['stopped'],pending=x['pending'],optimized_ms=statistics.median(x['total_ms']),unoptimized_same_fixture_ms=statistics.median(baseline['total_ms']),optimized_vs_normal_ratio=statistics.median(x['total_ms'])/statistics.median(normal['total_ms']),physical_equivalence=x['physical_digest']==baseline['physical_digest']))
 write('extra_controls',extras)
 write('control_summary',agg)
 stages=[]
 for case,t in sorted(set((r['case'],r['time']) for r in gpu)):
  selected=[r for r in gpu if r['case']==case and r['time']==t];total=sum(r['duration_ms'] for r in selected)
  for label in sorted(set(r['stage'] for r in selected)):stages.append(dict(case=case,time=t,stage=label,duration_ms=sum(r['duration_ms'] for r in selected if r['stage']==label),profiled_kernel_time_share=sum(r['duration_ms'] for r in selected if r['stage']==label)/total))
 write('kernel_stages',stages)
 correlations=[]
 def ranks(values):
  values=np.asarray(values);out=np.empty(len(values));order=np.argsort(values);k=0
  while k<len(order):
   end=k+1
   while end<len(order) and values[order[end]]==values[order[k]]:end+=1
   out[order[k:end]]=(k+end-1)/2;k=end
  return out
 for group in ['all','512','2048']:
  for label in sorted(set(r['stage'] for r in gpu)):
   matched=[]
   for r in gpu:
    if r['stage']!=label:continue
    w=next(x for x in whole if x['case']==r['case'] and x['time']==r['time'])
    if group!='all' and w['requested']!=int(group):continue
    matched.append((r,w))
   for feature in ['requested','active','stopped','connector','pending','requests','queried_movements','occupied_lanes','conflict_edges']:
    for metric in ['duration_ms','long_scoreboard','eligible','occupancy']:
     usable=[(r,w) for r,w in matched if math.isfinite(r[metric]) and (metric not in ['long_scoreboard','occupancy'] or 0<=r[metric]<=100)]
     x=ranks([w[feature] for r,w in usable]);y=ranks([r[metric] for r,w in usable]);rho=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else float('nan')
     correlations.append(dict(population_group=group,stage=label,feature=feature,metric=metric,windows=len(usable),spearman=rho))
 write('state_counter_correlations',correlations)

 source_rows=[]
 for path in (A/'profiles').glob('*/*/source.csv'):
  header=None;kernel='';filename=''
  for cells in csv.reader(path.open()):
   if cells and cells[0]=='Function Name':kernel=cells[1]
   elif cells and cells[0]=='File Path':filename=cells[1]
   elif 'stall_long_sb' in cells and 'Line No' in cells:header=cells
   elif header and len(cells)==len(header):
    row=dict(zip(header,cells))
    if row['Line No'].isdigit() and row['Address']=='-':
     value=row['stall_long_sb'].replace(',','').replace('-','')
     if value:source_rows.append(dict(case=path.parent.parent.name,time=path.parent.name,kernel=kernel,stage=stage(kernel),file=filename,line=row['Line No'],source=cells[1],long_scoreboard_samples=float(value)))
 write('source_samples',source_rows)
 retries=[]
 for path in (A/'profiles').glob('*/*/occupancy_retry/metrics.csv'):
  for row in list(csv.DictReader(path.open()))[1:]:retries.append(dict(case=path.parents[2].name,time=path.parents[1].name,kernel=row['Kernel Name'],occupancy=float(row['sm__warps_active.avg.pct_of_peak_sustained_active']),protocol='Separate occupancy-only retry; cache-control all'))
 write('occupancy_retries',retries)
 quality=[r for r in gpu+sgpu if r['quality_flags']];(O/'counter_quality.json').write_text(json.dumps(quality,indent=2))
 notes=['State selectors: 16=moving road interiors, 17=stopped road vehicles, 18=within 30 m of stop line, 19=connectors, 20=pending, 21=completed. Selectors 19–21 do not execute road following; their motion-kernel costs primarily represent record copying.','Table labels: D=distributed, C=concentrated; s1/s2=seeds 1401/1402; 72=signal controller, AP=always-permitted (not necessarily free flow).','Optimized backend frozen at 4ab1d4e, LPSIM_MOTION_CONTEXT=3. Original milestones unchanged.','48 full runs: 12 trajectory runs and 36 unprofiled timing repeats. Same demand, seeds, dt=0.25 s and 600 s horizon as the original study.','Controlled comparisons replay identical original snapshots; exact result comparisons cover all variants and state/region filters. Index construction is included in total replay times.','All-kernel Nsight samples restart one complete step, with cache-control all; they include leader sorting, context building and junction processing. Kernel sums exclude launch gaps and host work and are not unprofiled whole-step latency. Do not compare their percentages directly with earlier cache-control-none profiles.','Observed-state predictors are diagnostic; pre-run priors use only demand, topology and known signals. Geographic holdout is regions 12–15; seed 1402 is held out. Ridge remains fixed at 0.01.','State-counter correlations are descriptive Spearman associations across matched windows, also stratified by requested population; small samples and shared controller/topology effects prevent causal interpretation.','Whole-step predictors use median matched-step CUDA event times over three unprofiled repeats, with only 18 train/18 test windows on one network; these are exploratory and do not establish geographic transfer.','Regional filters retain full-state context and global index construction. Regional motion times are not additive regional execution costs. No allocator runtime speedup is established by these regression scores.','Near downstream occupied means the next receiving lane has a road vehicle within 14.5 m, not a unique causal label for denial. Vehicle-state categories overlap.','Out-of-range counter values remain in raw CSVs with quality flags; tables mark them Unreliable. Separate occupancy-only retries are reported without silently replacing primary samples.','Source instrumentation counts actual motion-loop visits separately from timing. Receiving-loop counts should be zero; conflict CSR traversal now belongs to shared construction.','A static allocator is gated on useful held-out improvement and a demonstrated schedulable workload; a positive regression result alone is insufficient.']
 score={(r['split'],r['model']):r for r in scores}
 findings=[f"Across 12 scenarios, the median optimized/original full-run GPU-time ratio is {statistics.median(r['optimized_gpu_ms']/r['old_gpu_ms'] for r in runs):.3f}; compare historical timing environments, not profiler percentages, for this ratio.",
 f"On the held-out seed, requested-count-only regional prediction has R2 {score['held_out_seed','requested_only']['r2']:.4f}; active-count MAE is {score['held_out_seed','active_count']['mae_ms']:.6f} ms and observed-work MAE is {score['held_out_seed','observed_work']['mae_ms']:.6f} ms.",
 f"On held-out seed and geography, active-count MAE is {score['held_out_seed_and_regions','active_count']['mae_ms']:.6f} ms and observed-work MAE is {score['held_out_seed_and_regions','observed_work']['mae_ms']:.6f} ms. High overall R2 can reflect the common all-record copy/launch cost rather than independent regional traffic work.",
 f"The pre-run state prior changes held-out-seed MAE by {100*(1-score['held_out_seed','prior_states']['mae_ms']/score['held_out_seed','prior_active']['mae_ms']):.2f}% versus the pre-run active-count prior; the geographic-holdout change is {100*(1-score['held_out_seed_and_regions','prior_states']['mae_ms']/score['held_out_seed_and_regions','prior_active']['mae_ms']):.2f}%."]
 for case in ['distributed_2048_seed1401_mode0','concentrated_2048_seed1401_mode0']:
  ss=sorted([r for r in stages if r['case']==case and r['time']==180],key=lambda r:-r['duration_ms'])
  findings.append(case+' at 180 s: '+', '.join(f"{r['stage']} {100*r['profiled_kernel_time_share']:.1f}%" for r in ss[:3])+' of summed profiled kernel duration. This excludes host/launch gaps and uses cold-cache replay.')
  sr=[r for r in source_rows if r['case']==case and r['stage']=='Junction arbitration'];total=sum(r['long_scoreboard_samples'] for r in sr)
  if sr:
   top=max(sr,key=lambda r:r['long_scoreboard_samples']);findings.append(f"Arbitration source samples for {case}: {100*top['long_scoreboard_samples']/max(1,total):.1f}% of its mapped long-scoreboard samples occur at line {top['line']}: {top['source']}")
 fullscores={(r['target'],r['model']):r for r in whole_scores}
 for target in ['full_step_ms','junction_ms']:
  b=fullscores[target,'active_count'];w=fullscores[target,'state_work'];findings.append(f"Observed state/work features reduce held-out-seed {target} MAE from {b['mae_ms']:.6f} to {w['mae_ms']:.6f} ms ({100*(1-w['mae_ms']/b['mae_ms']):.1f}%); R2 is {w['r2']:.4f}. This is a diagnostic oracle with only 18 training and 18 test windows on one network, not a pre-run regional forecast or causal test.")
 findings.append('No allocator runtime benefit is demonstrated. Regional replay work retains global context and copy overhead; the dominant whole-step sorting/arbitration work must be accounted for before treating regional predictions as schedulable cost.')
 (O/'conclusions.md').write_text('# Findings\n\n'+'\n\n'.join(findings)+'\n')
 notes=findings+notes
 with PdfPages(O/'optimized-state-study.pdf') as pdf:
  def table(title,headers,data,name):
   fig,ax=plt.subplots(figsize=(16,9));ax.axis('off');ax.set_title(title,fontsize=14);tab=ax.table(cellText=data,colLabels=headers,loc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.6);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/(name+'.png'),dpi=130);plt.close(fig)
  fig=plt.figure(figsize=(11.7,16.5));import textwrap;y=.96
  for note in notes:
   wrapped=textwrap.fill(note,95);height=.026*(wrapped.count('\n')+2)
   if y-height<.05:pdf.savefig(fig);plt.close(fig);fig=plt.figure(figsize=(11.7,16.5));y=.96
   fig.text(.06,y,wrapped,va='top',fontsize=11);y-=height
  pdf.savefig(fig);plt.close(fig)
  table('Full-run GPU time (3 unprofiled repeats)', ['Case','Completed','Pending','Old ms','Optimized ms','Ratio'],[[old.shortcase(r['case']),r['completed'],r['pending'],f"{r['old_gpu_ms']:.1f}",f"{r['optimized_gpu_ms']:.1f}",f"{r['optimized_gpu_ms']/r['old_gpu_ms']:.3f}"] for r in runs],'runs')
  for time in (60,180,450):
   selected=[r for r in windows if float(r['time'])==time and 'seed1401' in r['case']]
   table(f'Matched populations at {time} s, seed 1401 (categories can overlap)', ['Case','Pending','Stopped','Moving','Connector','Completed','Requests','Near leaders','Queued followers','Near blocked'],[[old.shortcase(r['case']),r['pending'],r['road_stopped'],r['road_moving'],r['connector'],r['completed'],r['requests'],r['approaching_leaders'],r['queued_followers'],r['near_downstream_occupied']] for r in selected],f'populations_{time}')
  table('Controlled factors: variant / reference total replay time',['Control','Old median ratio','Optimized median ratio'],[[r['control'],f"{r['old_ratio']:.3f}",f"{r['optimized_ratio']:.3f}"] for r in agg],'controls')
  if extras:
   table('Additional controlled fixtures; exact baseline/optimized results', ['Control','Median optimized ms','Median vs optimized normal'],[[label,f"{statistics.median(r['optimized_ms'] for r in extras if r['control']==label):.4f}",f"{statistics.median(r['optimized_vs_normal_ratio'] for r in extras if r['control']==label):.3f}"] for label in sorted(set(r['control'] for r in extras))],'extra_controls')
  table('Held-out regional motion prediction (diagnostic target)',['Holdout','Model','MAE ms','R2'],[[r['split'],r['model'],f"{r['mae_ms']:.5f}",f"{r['r2']:.3f}"] for r in scores],'predictions')
  table('Held-out seed: actual step and component times (18 train / 18 test windows)', ['Target','Model','MAE ms','R2'],[[r['target'],r['model'],f"{r['mae_ms']:.5f}",f"{r['r2']:.3f}"] for r in whole_scores],'whole_step_predictions')
  for name in ['distributed_2048_seed1401_mode0','concentrated_2048_seed1401_mode0']:
   selected=[r for r in gpu if r['case']==name and r['time']==180]
   table(name+' at 180 s: every kernel',['Stage','ms','Occ %','Eligible','Threads','Issue %','Long SB %','Load sectors'],[[r['stage'],f"{r['duration_ms']:.4f}",pct(r['occupancy']),f"{r['eligible']:.3f}",f"{r['threads']:.1f}",pct(r['issue']),pct(r['long_scoreboard']),f"{r['global_load_sectors']:.0f}"] for r in selected],name)
   table(name+' at 180 s: instruction and memory work',['Stage','Instructions','Global sectors','Local sectors','Registers/thread','SM %','L1/TEX %','L2 %'],[[r['stage'],f"{r['instructions']:.0f}",f"{r['global_load_sectors']:.0f}",f"{r['local_load_sectors']:.0f}",f"{r['registers']:.0f}",pct(r['sm_throughput']),pct(r['l1_throughput']),pct(r['l2_throughput'])] for r in selected],name+'_memory')
  table('State-filtered motion counters; filters retain all-record copying', ['Case','State selector','ms','Occ %','Eligible','Threads','Long SB %'],[[old.shortcase(r['case']),r['selector'],f"{r['duration_ms']:.4f}",pct(r['occupancy']),f"{r['eligible']:.3f}",f"{r['threads']:.1f}",pct(r['long_scoreboard'])] for r in sgpu],'state_gpu')
 (O/'README.md').write_text('# Optimized Berkeley state-cost study\n\n'+'\n\n'.join(notes)+'\n\nSee CSVs for every matched window, controls, hardware counters, held-out scores and kernel breakdown.\n')
 (O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Optimized Berkeley study</title><h1>Optimized Berkeley state-cost study</h1><p><a href="optimized-state-study.pdf">PDF</a> | <a href="README.md">Methods and limitations</a> | <a href="conclusions.md">Findings</a> | <a href="gpu_metrics.csv">GPU counters</a> | <a href="prediction_scores.csv">Held-out scores</a></p>'+''.join('<img style="width:100%" src="'+p.name+'">' for p in sorted(O.glob('*.png'))))
 print(json.dumps(dict(runs=len(runs),pipeline_samples=len(gpu),state_samples=len(sgpu),scores=scores,controls=agg),indent=2))
if __name__=='__main__':main()
