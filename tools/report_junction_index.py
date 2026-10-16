#!/usr/bin/env python3
import csv,json,statistics,math,html,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-junction-index-20261016';B=R/'artifacts/current/berkeley-optimized-state-20261016';O=R/'reports/berkeley-junction-index-20261016'
def read(p):return json.loads(p.read_text())
def write(name,rows):
 if not rows:return
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def fit(train,features):
 X=np.array([[1.]+[r[k] for k in features] for r in train]);y=np.array([r['arbitration_ms'] for r in train]);scale=np.maximum(np.std(X,axis=0),1.);Z=X/scale;penalty=np.eye(Z.shape[1])*.01;penalty[0,0]=0;beta=np.linalg.solve(Z.T@Z+penalty,Z.T@y)/scale
 return lambda r:max(0.,float(np.dot([1.]+[r[k] for k in features],beta)))
def main():
 O.mkdir(exist_ok=True);cases=read(B/'prepared.json')['cases'];local=[];runs=[];gpu=[];sources=[]
 for c in cases:
  name=c['name'];new=[read(A/'runs'/name/str(k)/'metrics.json') for k in (1,2,3)];old=[read(B/'runs'/name/str(k)/'metrics.json') for k in (1,2,3)]
  runs.append(dict(case=name,indexed_gpu_ms=statistics.median(x['advance_gpu_ms']+x['junction_gpu_ms'] for x in new),baseline_gpu_ms=statistics.median(x['advance_gpu_ms']+x['junction_gpu_ms'] for x in old),indexed_junction_ms=statistics.median(x['junction_gpu_ms'] for x in new),baseline_junction_ms=statistics.median(x['junction_gpu_ms'] for x in old),completed=new[0]['completed'],pending=new[0]['pending']))
  for t in (60,180,450):
   d=A/'local'/name/str(t);rows=list(csv.DictReader((d/'timings.csv').open()));features=[{k:int(v) for k,v in r.items()} for r in csv.DictReader((d/'timings.csv.junctions.csv').open())]
   for sel in range(-2,17):
    rr=[r for r in rows if int(r['selector'])==sel];f=[r for r in features if sel<0 or sel==16 or r['region']==sel]
    local.append(dict(case=name,seed=c['seed'],time=t,selector=sel,junctions=int(rr[0]['junctions']),requests=int(rr[0]['requests']),index_ms=statistics.median(float(r['index_ms']) for r in rr),arbitration_ms=statistics.median(float(r['arbitration_ms']) for r in rr),total_ms=statistics.median(float(r['total_ms']) for r in rr),arbitration_p10_ms=float(np.quantile([float(r['arbitration_ms']) for r in rr],.1)),arbitration_p90_ms=float(np.quantile([float(r['arbitration_ms']) for r in rr],.9)),green=sum(x['green'] for x in f),right_on_red=sum(x['right_on_red'] for x in f),preblocked=sum(x['preblocked'] for x in f),conflict_edges=sum(x['conflict_edges'] for x in f),max_requests=max([x['requests'] for x in f] or [0]),admitted=sum(x['admitted'] for x in f),denied=sum(x['denied'] for x in f)))
 for path in (A/'profiles').glob('*/*/*/metrics.csv'):
  raw=list(csv.DictReader(path.open()));units=raw[0]
  for x in raw[1:]:
   r=dict(case=path.parents[2].name,time=int(path.parents[1].name),variant=int(path.parent.name),kernel=x['Kernel Name'])
   for key,metric in [('duration_ms','gpu__time_duration.sum'),('occupancy','sm__warps_active.avg.pct_of_peak_sustained_active'),('eligible','smsp__warps_eligible.avg.per_cycle_active'),('threads','smsp__thread_inst_executed_per_inst_executed.ratio'),('long_scoreboard','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),('instructions','smsp__inst_executed.sum'),('global_load_sectors','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),('local_load_sectors','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),('registers','launch__registers_per_thread'),('sm_throughput','sm__throughput.avg.pct_of_peak_sustained_elapsed'),('issue_active','smsp__issue_active.avg.pct_of_peak_sustained_active'),('wait_stall','smsp__warp_issue_stalled_wait_per_warp_active.pct'),('not_selected','smsp__warp_issue_stalled_not_selected_per_warp_active.pct'),('uniform_branches','smsp__sass_average_branch_targets_threads_uniform.pct'),('l1_throughput','l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),('l2_throughput','lts__throughput.avg.pct_of_peak_sustained_elapsed')]:
    value=x.get(metric,'').replace(',','');r[key]=float(value) if value not in ('','n/a','N/A') else float('nan')
   r['duration_ms']*={'ns':1e-6,'us':1e-3,'µs':1e-3,'ms':1,'s':1000}[units['gpu__time_duration.sum']];r['quality_flags']=','.join(k for k in ['occupancy','long_scoreboard'] if not 0<=r[k]<=100);gpu.append(r)
 for path in (A/'profiles').glob('*/*/*/source.csv'):
  header=None;kernel=''
  for cells in csv.reader(path.open()):
   if cells and cells[0]=='Function Name':kernel=cells[1]
   elif 'stall_long_sb' in cells and 'Line No' in cells:header=cells
   elif header and len(cells)==len(header):
    x=dict(zip(header,cells))
    if x['Line No'].isdigit() and x['Address']=='-' and x['stall_long_sb'] not in ('','-'):sources.append(dict(case=path.parents[2].name,time=int(path.parents[1].name),variant=int(path.parent.name),kernel=kernel,line=x['Line No'],source=cells[1],long_scoreboard_samples=float(x['stall_long_sb'].replace(',',''))))
 # Only nonempty real local launches. Empty subsets have no kernel, not zero-cost simulated junctions.
 data=[r for r in local if 0<=r['selector']<16 and r['requests']>0];scores=[]
 models={'constant':[],'junction_count':['junctions'],'request_count':['requests'],'counts':['junctions','requests'],'state_work':['junctions','requests','right_on_red','preblocked','conflict_edges','max_requests']}
 for split in ['seed','seed_regions']:
  train=[r for r in data if r['seed']==1401 and (split=='seed' or r['selector']<12)];test=[r for r in data if r['seed']==1402 and (split=='seed' or r['selector']>=12)];y=np.array([r['arbitration_ms'] for r in test])
  for name,features in models.items():
   f=fit(train,features);p=np.array([f(r) for r in test]);scores.append(dict(split=split,model=name,train=len(train),test=len(test),mae_ms=float(np.mean(abs(p-y))),r2=float(1-np.sum((y-p)**2)/max(1e-20,np.sum((y-y.mean())**2)))))
 comparisons=[]
 for c in cases:
  for t in (60,180,450):
   rs={r['selector']:r for r in local if r['case']==c['name'] and r['time']==t};old,idx,compact=rs[-2],rs[-1],rs[16]
   comparisons.append(dict(case=c['name'],time=t,requests=idx['requests'],active_junctions=idx['junctions'],reference_ms=old['total_ms'],indexed_total_ms=idx['total_ms'],index_ms=idx['index_ms'],indexed_arbitration_ms=idx['arbitration_ms'],compact_arbitration_ms=compact['arbitration_ms'],local_sum_ms=sum(rs[k]['arbitration_ms'] for k in range(16) if rs[k]['requests']),ratio=idx['total_ms']/old['total_ms']))
 for name,rows in [('runs',runs),('local_work',local),('comparisons',comparisons),('prediction_scores',scores),('gpu_metrics',gpu),('source_samples',sources)]:write(name,rows)
 findings=[f"Median full-run GPU-time ratio versus the previous optimized baseline: {statistics.median(r['indexed_gpu_ms']/r['baseline_gpu_ms'] for r in runs):.3f}. Full-run timings include request-index construction.",f"Median isolated indexed arbitration plus construction / full-scan arbitration ratio: {statistics.median(r['ratio'] for r in comparisons):.3f} across 36 matched snapshots.",f"Nonempty regional arbitration launches have median duration {statistics.median(r['arbitration_ms'] for r in data)*1000:.2f} microseconds. Shared index construction is measured separately, not assigned to each region."]
 notes=['Opt-in LPSIM_JUNCTION_REQUEST_INDEX=1. Sort keys encode (junction, green-before-red pass, stable vehicle ID); invalid requests sort to the end. Blocked requests remain in their original priority positions. Admission logic is shared with the full-scan reference.','Validation covers following and exclusive modes, exact request membership/priority, car state, reservation summaries and complete Berkeley trajectories. Local physical results are compared to full arbitration; unselected junctions must remain unchanged.','Local replay prepares immutable post-motion requests/context, then restores input cars/summaries outside timing. Timed regional launches touch only selected active junctions and their request ranges. There is no full-car copy inside the measured interval. Global index construction is included for selector -1 and in full simulations.','Selectors: -2=full-scan reference; -1=indexed full grid plus construction; 16=compact active-junction launch with prebuilt index; 0–15=local regional groups with prebuilt index. Local group assignment uses the first incoming movement’s road region. Empty subsets have no kernel launch and are excluded from fitting.','Local preparation, host-to-device resets and host-built junction lists are excluded from diagnostic arbitration timings. They are not deployable allocator timings. Regional launches change batching/occupancy, so summing their times is not a full-kernel cost decomposition.','Full-run reference timings are historical from the frozen preceding study; local replays and Nsight before/after samples use the same current binary. Three warmups and ten timed samples per immutable local workload. Three complete-run timing repeats per scenario. Nsight is separate, cache-control all, matched single-step restarts. Out-of-range percentage counters are flagged and not interpreted.','Prediction uses only pre-arbitration features; admitted/denied outcomes are recorded for diagnosis but excluded from models. Fixed ridge 0.01; seed 1402 and regions 12–15 are holdouts. State predictors see observed requests, not a pre-run forecast. No allocator is implemented or speedup inferred from prediction alone.']
 for label in ['distributed','concentrated']:
  pair={r['variant']:r for r in gpu if r['case']==label+'_2048_seed1401_mode0' and r['time']==180 and 'arbitrate' in r['kernel']}
  if len(pair)==2:
   x,y=pair[0],pair[1];ss={v:sum(r['long_scoreboard_samples'] for r in sources if r['case']==label+'_2048_seed1401_mode0' and r['variant']==v) for v in [0,1]}
   findings.append(f"{label}, 180 s: cold-cache arbitration {x['duration_ms']*1000:.3f} -> {y['duration_ms']*1000:.3f} us; long-scoreboard fraction {x['long_scoreboard']:.2f}% -> {y['long_scoreboard']:.2f}%; mapped long-scoreboard samples {ss[0]:.0f} -> {ss[1]:.0f}. Percentages describe the remaining execution, not total waiting time. Samples are single-launch observations, not confidence intervals.")
 for split in ['seed','seed_regions']:
  ss={r['model']:r for r in scores if r['split']==split}
  findings.append(f"Local {split} holdout: constant / counts / state-work MAE = {ss['constant']['mae_ms']*1000:.2f} / {ss['counts']['mae_ms']*1000:.2f} / {ss['state_work']['mae_ms']*1000:.2f} us. These are observed-state diagnostic predictors; no pre-run allocator benefit has been demonstrated.")
 findings.append('Richer state features add little beyond junction/request counts: seed-holdout MAE improves by only about 0.08 us, and the geographic holdout has only 19 observations and negative R2. This is insufficient evidence for a state-aware allocator advantage. First evaluate cheaper batching/index construction and the remaining leader sort; retain counts as the allocator baseline.')
 for label in ['distributed','concentrated']:
  rr=[r for r in gpu if r['case']==label+'_2048_seed1401_mode0' and r['time']==180 and r['variant']==1];total=sum(r['duration_ms'] for r in rr);leader=sum(r['duration_ms'] for r in rr if 'DeviceMergeSort' in r['kernel'])
  findings.append(f"{label}, 180 s: leader-index sorting accounts for {100*leader/total:.1f}% of the indexed complete-step kernel-duration sum. This cold-cache kernel sum excludes host work, allocations and gaps; it is not unprofiled elapsed step time.")
 notes.append('Reproduction: compile src/simulator/scalable_junction_simulator.cu and tools/junction_local_replay.cu with nvcc -O3 -lineinfo -arch=sm_87 -std=c++17; preserve binaries and source hashes. Use a fresh writable A directory in experiment_junction_index.py, place simulator and local_replay there, then run the driver. It requires passwordless sudo for fixed clocks/Nsight, restores clocks in finally, and references frozen inputs in the preceding study. The archived run must not be overwritten.')
 notes.append('Indexed admission removes the reference 2 * junctions * requested-vehicles request visits; each valid request is visited once in arbitration. Index construction still reads and sorts all requested-vehicle records. One thread still serializes admissions within its junction; different junctions run concurrently. Vehicle-ID priority is the unchanged vehicle-array index in this implementation; reordering vehicle storage would require an explicit persistent priority-ID mapping.')
 (O/'conclusions.md').write_text('# Junction indexing results\n\n'+'\n\n'.join(findings)+'\n')
 (O/'README.md').write_text('# Methods\n\n'+'\n\n'.join(notes)+'\n')
 with PdfPages(O/'junction-index.pdf') as pdf:
  fig=plt.figure(figsize=(11.7,16.5));y=.95
  for note in findings+notes:
   text=textwrap.fill(note,95);height=.025*(text.count('\n')+2)
   if y-height<.05:pdf.savefig(fig);plt.close(fig);fig=plt.figure(figsize=(11.7,16.5));y=.95
   fig.text(.06,y,text,va='top',fontsize=11);y-=height
  pdf.savefig(fig);plt.close(fig)
  def table(title,cols,rows,name):
   fig,ax=plt.subplots(figsize=(16,9));ax.axis('off');ax.set_title(title);t=ax.table(cellText=rows,colLabels=cols,loc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.7);
   if cols[0]=='Case':
    for (row,col),cell in t.get_celld().items():
     if col==0:cell.set_fontsize(6)
   fig.tight_layout();pdf.savefig(fig);fig.savefig(O/(name+'.png'),dpi=130);plt.close(fig)
  table('Full runs: request-index construction included',['Case','Old GPU ms','Indexed GPU ms','Ratio','Completed'],[[r['case'],f"{r['baseline_gpu_ms']:.1f}",f"{r['indexed_gpu_ms']:.1f}",f"{r['indexed_gpu_ms']/r['baseline_gpu_ms']:.3f}",r['completed']] for r in runs],'runs')
  table('Local work, held-out prediction: nonempty regions only',['Holdout','Model','Train','Test','MAE us','R2'],[[r['split'],r['model'],r['train'],r['test'],f"{1000*r['mae_ms']:.3f}",f"{r['r2']:.3f}"] for r in scores],'predictions')
  table('180 s arbitration: matched immutable input',['Case','Requests','Junctions','Scan ms','Index build ms','Indexed arb ms','Compact arb ms'],[[r['case'],r['requests'],r['active_junctions'],f"{r['reference_ms']:.4f}",f"{r['index_ms']:.4f}",f"{r['indexed_arbitration_ms']:.4f}",f"{r['compact_arbitration_ms']:.4f}"] for r in comparisons if r['time']==180],'local')
  source_rows=[]
  for label in ['distributed','concentrated']:
   for variant in [0,1]:
    rr=[r for r in sources if r['case']==label+'_2048_seed1401_mode0' and r['variant']==variant];total=sum(r['long_scoreboard_samples'] for r in rr)
    source_rows.append([label,variant,f'{total:.0f}',', '.join(str(r['line'])+': '+str(int(r['long_scoreboard_samples'])) for r in sorted(rr,key=lambda x:x['long_scoreboard_samples'],reverse=True)[:3])])
  table('Mapped long-scoreboard samples, 180 s; source line : samples',['Case','Indexed','Total samples','Largest source locations'],source_rows,'source_samples')
  gr=[r for r in gpu if r['time']==180 and 'arbitrate' in r['kernel']]
  table('Arbitration GPU counters at matched 180 s windows',['Case','Indexed','ms','Long SB %','Occupancy %','Eligible','Threads','Instructions'],[[r['case'],r['variant'],f"{r['duration_ms']:.4f}",f"{r['long_scoreboard']:.2f}",f"{r['occupancy']:.2f}" if 0<=r['occupancy']<=100 else 'Unreliable',f"{r['eligible']:.3f}",f"{r['threads']:.1f}",f"{r['instructions']:.0f}"] for r in gr],'gpu')
  metric_labels=[('duration_ms','Kernel duration (ms)'),('sm_throughput','SM throughput (% peak)'),('occupancy','Achieved occupancy (%)'),('eligible','Eligible warps / scheduler cycle'),('threads','Active threads / instruction'),('issue_active','Issue-active cycles (%)'),('long_scoreboard','Long-scoreboard stall (%)'),('wait_stall','Fixed-latency wait stall (%)'),('not_selected','Eligible, not selected (%)'),('uniform_branches','Uniform branch targets (%)'),('l1_throughput','L1/TEX throughput (% peak)'),('l2_throughput','L2 throughput (% peak)'),('instructions','Warp instructions'),('global_load_sectors','Global load sectors'),('local_load_sectors','Local load sectors'),('registers','Registers / thread')]
  for label in ['distributed','concentrated']:
   rr={(r['time'],r['variant']):r for r in gpu if label+'_' in r['case'] and 'arbitrate' in r['kernel']}
   cells=[]
   for key,title in metric_labels:
    values=[]
    for time in [60,180,450]:
     values.append(' / '.join('Unreliable' if key in x['quality_flags'].split(',') else f"{x[key]:.3f}" for x in [rr[(time,0)],rr[(time,1)]]))
    cells.append([title]+values)
   table(label+' arbitration: full scan / request index',['Metric','60 s','180 s','450 s'],cells,'counters_'+label)
  fig,ax=plt.subplots(figsize=(10,6))
  for seed in [1401,1402]:
   rr=[r for r in data if r['seed']==seed];ax.scatter([r['requests'] for r in rr],[r['arbitration_ms']*1000 for r in rr],s=12,alpha=.5,label=str(seed))
  ax.set_xlabel('Requests in selected region');ax.set_ylabel('Arbitration microseconds (index already built)');ax.set_title('Actual local work; association is not causation');ax.legend();fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'local_scatter.png',dpi=130);plt.close(fig)
 (O/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Junction request indexing and local work</h1><p><a href="junction-index.pdf">PDF</a> | <a href="conclusions.md">Findings</a> | <a href="README.md">Methods</a> | <a href="gpu_metrics.csv">GPU counters</a> | <a href="local_work.csv">Local measurements</a></p>'+''.join('<img style="width:100%" src="'+p.name+'">' for p in sorted(O.glob('*.png'))))
 print(json.dumps(dict(findings=findings,scores=scores),indent=2))
if __name__=='__main__':main()
