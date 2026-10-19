#!/usr/bin/env python3
"""Report preregistered vehicle assignment test; no tuning from evaluation data."""
import csv,json,hashlib,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1]; A=R/'artifacts/current/vehicle-allocation-locality-20261018'; O=R/'reports/vehicle-allocation-locality-20261018'
NAMES=['Direct','Identity','Geographic','Vehicle count','State grouped','Contiguous grouped','Contiguous balanced']
def save(name,rows):
 with (O/(name+'.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def median(xs):return float(np.median(xs))
def main():
 assert json.loads((A/'complete.json').read_text())['passed'];O.mkdir(exist_ok=True)
 cases=json.loads((A/'cases.json').read_text()); rows=[];cost=[]
 for b in [1,2,3]:
  for c in cases:
   for t in [60,180,300,450]:
    d=A/'runs'/str(b)/c['name']/str(t);raw=list(csv.DictReader((d/'timings.csv').open()));assert len(raw)==140 and all(x['exact']=='true' for x in raw)
    cr=list(csv.DictReader((d/'timings.csv.costs.csv').open()));cost.append(dict(batch=b,case=c['name'],time=t,split=c['split'],requested=c['n'],fit_wall_ms=float(cr[0]['fit_wall_ms'])))
    for ch in [0,1]:
     for p in range(7):
      rr=[x for x in raw if int(x['policy'])==p and int(x['charged'])==ch];assert len(rr)==10
      row=dict(batch=b,case=c['name'],time=t,split=c['split'],pattern=c['pattern'],requested=c['n'],policy=p,name=NAMES[p],charged=ch)
      row.update({k:median([float(x[k]) for x in rr]) for k in ['pre_ms','setup_ms','motion_ms','junction_ms','pipeline_ms','wall_ms']});rows.append(row)
 lookup={(r['batch'],r['case'],r['time'],r['charged'],r['policy']):r for r in rows}
 for r in rows:
  for base,label in [(3,'count'),(0,'direct')]:
   ref=lookup[(r['batch'],r['case'],r['time'],r['charged'],base)]
   for m in ['motion_ms','wall_ms']:r[m+'_ratio_'+label]=r[m]/ref[m]
 save('snapshot_medians',rows);save('oracle_fitting',cost)
 summary=[]
 for split in ['train','validation','test']:
  # Input manifest uses val for validation in some versions.
  accepted={split,'val'} if split=='validation' else {split}
  for ch in [0,1]:
   for p in range(7):
    rr=[r for r in rows if r['split'] in accepted and r['charged']==ch and r['policy']==p]
    if not rr:continue
    row=dict(split=split,charged=ch,policy=p,name=NAMES[p],windows=len(rr)//3)
    for k in ['pre_ms','setup_ms','motion_ms','junction_ms','wall_ms','motion_ms_ratio_direct','motion_ms_ratio_count','wall_ms_ratio_count','wall_ms_ratio_direct']:row[k]=median([r[k] for r in rr])
    summary.append(row)
 save('policy_summary',summary)
 gate=[]
 for p in [1,2,3,4,5,6]:
  rr=[r for r in rows if r['split']=='test' and r['charged']==1 and r['policy']==p]
  batches=[median([r['wall_ms_ratio_direct'] for r in rr if r['batch']==b]) for b in [1,2,3]]
  keys=sorted({(r['case'],r['time']) for r in rr}); ratios=[median([r['wall_ms_ratio_direct'] for r in rr if (r['case'],r['time'])==key]) for key in keys]
  wins=sum(x<1 for x in ratios)/len(ratios)
  gate.append(dict(policy=p,name=NAMES[p],batch_ratios=batches,win_fraction=wins,windows=len(keys),exact=True,passed=all(x<=.95 for x in batches) and wins>=.75))
 (O/'gate.json').write_text(json.dumps(gate,indent=2))
 bysize=[]
 for n in [2048,8192,16384]:
  for ch in [0,1]:
   for p in range(7):
    rr=[r for r in rows if r['split']=='test' and r['requested']==n and r['charged']==ch and r['policy']==p]
    bysize.append(dict(requested=n,charged=ch,policy=p,name=NAMES[p],motion_ms=median([r['motion_ms'] for r in rr]),wall_ms=median([r['wall_ms'] for r in rr]),wall_ratio_direct=median([r['wall_ms_ratio_direct'] for r in rr]),wall_ratio_count=median([r['wall_ms_ratio_count'] for r in rr]),motion_ratio_count=median([r['motion_ms_ratio_count'] for r in rr])))
 save('demand_summary',bysize)
 patterns=[]
 for pattern in ['distributed','concentrated','mixed','bursty']:
  for ch in [0,1]:
   for p in range(7):
    rr=[r for r in rows if r['split']=='test' and r['pattern']==pattern and r['charged']==ch and r['policy']==p]
    patterns.append(dict(pattern=pattern,charged=ch,name=NAMES[p],wall_ratio_direct=median([r['wall_ms_ratio_direct'] for r in rr]),wall_ratio_count=median([r['wall_ms_ratio_count'] for r in rr]),motion_ratio_count=median([r['motion_ms_ratio_count'] for r in rr])))
 save('pattern_summary',patterns)
 prior=list(csv.DictReader((R/'reports/vehicle-allocation-oracle-20261018/snapshot_medians.csv').open()))
 old={(int(r['batch']),r['case'],int(r['time']),int(r['charged']),int(r['policy'])):r for r in prior}
 comparison=[]
 for ch in [0,1]:
  for p in [5,6]:
   rr=[r for r in rows if r['split']=='test' and r['charged']==ch and r['policy']==p]
   comparison.append(dict(charged=ch,policy=NAMES[p],old_step_ratio_direct=median([float(old[r['batch'],r['case'],r['time'],ch,p]['wall_ms_ratio_direct']) for r in rr]),new_step_ratio_direct=median([r['wall_ms_ratio_direct'] for r in rr]),old_motion_ratio_direct=median([float(old[r['batch'],r['case'],r['time'],ch,p]['motion_ms_ratio_direct']) for r in rr]),new_motion_ratio_direct=median([r['motion_ms_ratio_direct'] for r in rr])))
 save('prior_comparison',comparison)
 gpu=[]
 if (A/'profiles/complete.json').exists():
  for path in sorted((A/'profiles').glob('*/*/metrics.csv')):
   raw=list(csv.DictReader(path.open()));assert len(raw)==2,(path,len(raw));units,x=raw
   row=dict(case=path.parents[1].name,policy=int(path.parent.name),name=NAMES[int(path.parent.name)])
   for key,metric in [('duration_ms','gpu__time_duration.sum'),('occupancy','sm__warps_active.avg.pct_of_peak_sustained_active'),('long_scoreboard','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),('threads','smsp__thread_inst_executed_per_inst_executed.ratio'),('instructions','smsp__inst_executed.sum'),('global_load_sectors','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),('global_load_requests','l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum'),('l2_read_sectors','lts__t_sectors_op_read.sum'),('registers','launch__registers_per_thread')]:
    value=x.get(metric,'').replace(',','');row[key]=float(value) if value not in ('','n/a','N/A') else float('nan')
   row['duration_ms']*={'ns':1e-6,'us':1e-3,'µs':1e-3,'ms':1,'s':1000}[units['gpu__time_duration.sum']]
   row['sectors_per_request']=row['global_load_sectors']/row['global_load_requests']
   row['quality_flags']=','.join(k for k in ['occupancy','long_scoreboard'] if not 0<=row[k]<=100)
   gpu.append(row)
  assert len(gpu)==10
  save('gpu_metrics',gpu)
 passed=[g['name'] for g in gate if g['passed']]
 test={(r['charged'],r['policy']):r for r in summary if r['split']=='test'}
 notes=[
  f"Preserving original 32-ID groups removes the old assignment regression, but does not establish a speedup over the original. The earlier supplied grouped/balanced oracle step ratios were {comparison[0]['old_step_ratio_direct']:.3f}x / {comparison[1]['old_step_ratio_direct']:.3f}x original; the new ratios are {comparison[0]['new_step_ratio_direct']:.3f}x / {comparison[1]['new_step_ratio_direct']:.3f}x. Different experiments use concurrent direct timings for normalization. Tiny differences near 1 are not a practically meaningful improvement.",
  f"Supplied contiguous grouping: {test[0,5]['wall_ms_ratio_direct']:.4f}x original direct complete-step time. Supplied contiguous balancing: {test[0,6]['wall_ms_ratio_direct']:.4f}x direct. Ratios below 1 indicate improvement. Charged ratios are {test[1,5]['wall_ms_ratio_direct']:.3f}x and {test[1,6]['wall_ms_ratio_direct']:.3f}x, respectively.",
  'Preregistered charged gate vs ORIGINAL DIRECT: '+(', '.join(passed)+' passed.' if passed else 'No candidate passed.')+' Requires at least 5% median complete-step improvement in each of three batches, wins on at least 75% of 60 historical test snapshots, and exact results.',
  'Scope: 24 frozen Berkeley demand runs, 96 matched snapshots at 60/180/300/450 s; three process batches, three warmups and ten randomized timing repetitions per policy/cost mode. All 40,320 recorded evaluations passed vehicle-byte and semantic reservation-summary equivalence; motion comparisons and warmups also passed. These are one-step replays, not new full traffic trajectories.',
  'The original train/validation/test labels are retained for comparability. These snapshots have already been inspected in the earlier experiment; they are not newly unseen evaluation data. This follow-up tests a declared mechanism without fitting parameters to the new timings. One topology and simulator do not establish generalization.',
  'The oracle now uses contiguous 32-ID groups from ORIGINAL storage order. Five separate instrumented calibration passes measure contextual elapsed warp cycles. Grouping orders tiles by cost; balancing distributes tiles to blocks, four per block. Stable IDs and within-warp adjacency remain intact. Inter-warp locality can still change; secondary neighbor and connector accesses need not be contiguous. Tile membership also changes the mix of vehicle states within warps relative to the old striped oracle, so this is not a pure memory-locality intervention.',
  'The previous oracle used 32-ID groups from the scattered count schedule. Only this oracle membership/calibration changes in the new harness; direct, identity, geographic, count and state policies are concurrent controls. All policies execute the same vehicle calculations and memory structures. Full vehicle record storage is unchanged; this is task-list assignment, not state repacking.',
  'Supplied schedules exclude construction, transfer and fitting. Charged schedules rebuild each step: state download, CPU assignment, upload, synchronization and (for oracle candidates) five calibration launches. The inherited generic path downloads state even for identity and oracle schedules; these conservative charged results do not represent a minimal GPU-resident or amortized implementation. Persistent buffer allocation is excluded.',
  'Complete-step wall time includes shared leader indexes, motion context and junction work. Stage event intervals include host launch gaps and charged setup gaps. Medians of matched ratios are not ratios of aggregate medians. Reordering contextual costs is a heuristic measured-cost candidate, not a guaranteed optimal schedule.',
  'Separate Nsight diagnostics use distributed and concentrated 16,384-record snapshots at 300 s, with cold-cache replay of only the vehicle-update kernel. Single-launch counters are descriptive, not confidence intervals or complete-step timing. Long-scoreboard percentage is a fraction of warp cycles, not total waiting time. Out-of-range percentages are flagged. The original-order identity control separates task indirection overhead from reassignment effects. Hardware: Jetson Orin Nano, 15 W power mode, fixed 612 MHz GPU and 2133 MHz EMC during timing; original clock configuration restored.'
 ]
 if gpu:
  for case in sorted({r['case'] for r in gpu}):
   g={r['policy']:r for r in gpu if r['case']==case};base,count,group=g[0],g[3],g[5]
   notes.append(f"Nsight {case} at 300 s: direct/count/contiguous-grouped duration = {base['duration_ms']:.3f}/{count['duration_ms']:.3f}/{group['duration_ms']:.3f} ms; occupancy = {base['occupancy']:.1f}/{count['occupancy']:.1f}/{group['occupancy']:.1f}%; long-scoreboard fraction = {base['long_scoreboard']:.2f}/{count['long_scoreboard']:.2f}/{group['long_scoreboard']:.2f}%. Count assignment has {(count['l2_read_sectors']/base['l2_read_sectors']-1)*100:.1f}% more L2 read sectors than direct. The slower count assignment has higher occupancy and lower long-scoreboard percentage; neither metric alone predicts runtime. Contiguous grouping does not consistently improve direct's traffic or stalls.")
  populations=[r for r in csv.DictReader((R/'reports/berkeley-allocation-workloads-20261018/snapshots.csv').open()) if r['case'] in {g['case'] for g in gpu} and float(r['time'])==300]
  save('profile_populations',populations)
  notes.append('Counter population context: '+ '; '.join(f"{r['case']}: {r['road_moving']} road-moving, {r['road_stopped']} road-stopped, {r['connector']} connectors, {r['eligible_pending']} eligible pending, {r['future']} future departures, {r['completed']} completed" for r in populations)+'. Counters describe only the motion kernel; they do not describe the occupancy of shared index or junction kernels.')
 (O/'conclusions.md').write_text('\n\n'.join(notes));(O/'protocol.md').write_text((A/'protocol.md').read_text())
 with PdfPages(O/'vehicle-allocation-locality.pdf') as pdf:
  for start in range(0,len(notes),4):
   fig=plt.figure(figsize=(12,9));fig.text(.05,.95,'Vehicle-work allocation: contiguous original-order tiles',fontsize=17,va='top');fig.text(.05,.88,'\n\n'.join(textwrap.fill(s,112) for s in notes[start:start+4]),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  for ch in [0,1]:
   ss=[r for r in summary if r['split']=='test' and r['charged']==ch]
   fig,ax=plt.subplots(figsize=(14,6));ax.axis('off');ax.set_title(('Charged' if ch else 'Supplied schedule')+' — historical test snapshots; median per-window measurements')
   tab=ax.table(cellText=[[r['name'],f"{r['motion_ms']*1000:.1f}",f"{r['setup_ms']*1000:.1f}",f"{r['wall_ms']*1000:.1f}",f"{r['motion_ms_ratio_direct']:.3f}",f"{r['wall_ms_ratio_count']:.3f}",f"{r['wall_ms_ratio_direct']:.3f}"] for r in ss],colLabels=['Policy','Motion µs','Setup µs','Step wall µs','Motion/direct','Step/count','Step/direct'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'table_{ch}.png',dpi=160);plt.close(fig)
  for ch in [0,1]:
   ss=[r for r in summary if r['split']=='test' and r['charged']==ch]
   fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title(('Charged' if ch else 'Supplied schedule')+' stage breakdown — median milliseconds')
   tab=ax.table(cellText=[[r['name'],*[f'{r[k]:.3f}' for k in ['pre_ms','setup_ms','motion_ms','junction_ms','wall_ms']]] for r in ss],colLabels=['Policy','Shared indexes','Setup','Vehicle update','Junction pipeline','Step wall'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.2);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
  fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title('Previous scattered tiles vs new contiguous tiles — normalized to concurrent direct')
  tab=ax.table(cellText=[[r['policy'],'Charged' if r['charged'] else 'Supplied',f"{r['old_step_ratio_direct']:.3f}",f"{r['new_step_ratio_direct']:.3f}",f"{r['old_motion_ratio_direct']:.3f}",f"{r['new_motion_ratio_direct']:.3f}"] for r in comparison],colLabels=['Policy','Cost mode','Old step/direct','New step/direct','Old motion/direct','New motion/direct'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.3);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'prior_comparison.png',dpi=160);plt.close(fig)
  for case in sorted({r['case'] for r in gpu}):
   rr=sorted([r for r in gpu if r['case']==case],key=lambda r:r['policy'])
   fields=[('duration_ms','Profiled duration (ms)'),('occupancy','Achieved occupancy (%)'),('long_scoreboard','Long scoreboard (%)'),('threads','Threads / instruction'),('instructions','Warp instructions'),('global_load_sectors','Global-load sectors'),('global_load_requests','Global-load requests'),('sectors_per_request','Sectors / request'),('l2_read_sectors','L2 read sectors'),('registers','Registers / thread')]
   fig,ax=plt.subplots(figsize=(15,8));ax.axis('off');ax.set_title(case+' at 300 s — separate cold-cache Nsight sample')
   tab=ax.table(cellText=[[label]+[f"{r[k]:,.3f}" if k in ['duration_ms','occupancy','long_scoreboard','threads','sectors_per_request'] else f"{r[k]:,.0f}" for r in rr] for k,label in fields],colLabels=['Metric']+[r['name'] for r in rr],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.1);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/(case+'_gpu.png'),dpi=160);plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(14,6))
  for ch,ax in enumerate(axs):
   for p in range(1,7):
    rr=[r for r in bysize if r['charged']==ch and r['policy']==p];ax.plot([r['requested'] for r in rr],[r['wall_ratio_direct'] for r in rr],marker='o',label=NAMES[p])
   ax.axhline(1,color='black',lw=1);ax.set_title('Charged' if ch else 'Supplied schedule');ax.set_xlabel('Requested records');ax.set_ylabel('Complete-step wall / original direct');ax.set_xscale('log',base=2);ax.legend(fontsize=8)
  fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'step_ratios.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title('Preregistered gate: charged complete step vs original direct implementation')
  tab=ax.table(cellText=[[g['name'],*[f'{x:.3f}' for x in g['batch_ratios']],f"{g['win_fraction']*100:.1f}%",'PASS' if g['passed'] else 'FAIL'] for g in gate],colLabels=['Policy','Batch 1 ratio','Batch 2 ratio','Batch 3 ratio','Snapshot wins','Gate'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.3);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'gate.png',dpi=160);plt.close(fig)
 print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
