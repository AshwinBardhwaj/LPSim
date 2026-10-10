#!/usr/bin/env python3
import csv,json,statistics,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from milestone2_snapshots import ART,ROOT,dump
from profile_junction_experiment import METRICS
OUT=ROOT/'reports/milestone2-snapshots-20261009'
LABELS=['Kernel duration (µs)','L1/TEX throughput (% peak)','L2 throughput (% peak)','Uniform branch targets (%)','Fixed-latency wait (% warp cycles)','Eligible, not selected (% warp cycles)','SM throughput (% peak)','Achieved occupancy (%)','Eligible warps / scheduler cycle','Active threads / instruction (of 32)','Issue-active cycles (%)','Long-scoreboard stalls (% warp cycles)']
ORDER=[0,6,7,8,9,10,11,4,5,3,1,2]
EXTRA=['smsp__inst_executed.sum','smsp__thread_inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','lts__t_sectors_op_read.sum','launch__registers_per_thread','launch__local_mem_per_thread','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
EXTRALABEL=['Warp instructions executed','Thread instructions executed','L1 global-load sectors','L2 read sectors','Registers per thread','Local memory per thread (bytes)','L1 local-load sectors','L1 local-store sectors']
def main():
 assert json.loads((ART/'validation.json').read_text())['passed'];OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
 cfg=json.loads((ART/'manifest.json').read_text());data=[]
 for c in cfg['cases']:
  name=c['name'];timings=[json.loads((ART/'runs'/name/f'{r}.csv.timing.json').read_text())['milliseconds'] for r in range(3)];med=[statistics.median(t)*1000 for t in timings];r=dict(c,time_us=statistics.median(med),repeat_medians_us=med,min_us=min(med),max_us=max(med),us_per_road_active=statistics.median(med)/c['active'])
  prof=list(csv.DictReader((ART/'profiles'/name/'metrics.csv').open()));unit=prof[0];p=prof[1]
  for k in METRICS+EXTRA:
   try:r[k]=float(p[k].replace(',',''))
   except (KeyError,ValueError):r[k]=None
  r[METRICS[0]]*=dict(ns=.001,us=1,ms=1000,s=1e6)[unit[METRICS[0]]]
  r.update(json.loads((ART/'runs'/name/'instrumented.csv.counts.json').read_text()));data.append(r)
 dump(OUT/'summary.json',data)
 with (OUT/'metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 source=[]
 for name in ('motion_50','padding_pending_8192'):
  rows=list(csv.reader((ART/'source_profiles'/name/'source.csv').open()));h=rows[2];idx=h.index('stall_long_sb');records=[dict(line=int(r[0]),code=r[1],samples=float(r[idx].replace(',',''))) for r in rows[3:] if r and r[0].isdigit() and r[2]=='-'];total=sum(r['samples'] for r in records)
  for r in records:r.update(case=name,share_percent=100*r['samples']/total if total else 0)
  source+=sorted(records,key=lambda r:r['samples'],reverse=True)
 dump(OUT/'source-stalls.json',source)
 lookup={r['name']:r for r in data};base=lookup['motion_50'];pad=lookup['padding_pending_8192'];motion=[lookup[f'motion_{n}'] for n in (0,50,90,100)]
 motion_text=', '.join(format(r['time_us'],'.1f') for r in motion)
 findings=[f"All 15 fixtures pass spacing, repeat, instrumented/uninstrumented and Nsight-output checks. Padding and storage permutations preserve active physical results exactly.",f"Movement: 0/50/90/100% stopped yield {motion_text} µs/step with identical requested and active counts (2,048).",f"Inactive padding: holding 2,048 physical road vehicles fixed, 8,192 total records cost {pad['time_us']/base['time_us']:.2f}× the unpadded step ({base['time_us']:.1f} → {pad['time_us']:.1f} µs).",f"The instrumented build counts A×N receiving checks, A×(N−1) leader candidates and A×N connector checks—even with no connector occupants.",f"Shuffling storage: {lookup['padding_pending_4096']['time_us']:.1f} → {lookup['order_shuffle']['time_us']:.1f} µs, despite fewer active threads/instruction. Eligible warps increase. Grouping, access order and scheduling all change.","Source-correlated long-scoreboard samples identify scan-loop locations. A sampled stalled instruction is not, by itself, proof of the exact upstream load that caused its dependency.","Leader indexing is a justified next experiment, but it removes only one of three full-array scans. Receiving-lane and connector searches must be measured separately.","These are one-step road-kernel fixtures, not calibrated traffic or a full Berkeley junction simulation. All stop-line request counts are zero by construction."]
 (OUT/'conclusions.md').write_text('# Milestone 2A conclusions\n\n'+'\n\n'.join(findings)+'\n')
 pages=[]
 with PdfPages(OUT/'milestone2-snapshots.pdf') as pdf:
  def save(fig,name,caption=''):
   fig.text(.04,.024,textwrap.fill(caption,180),fontsize=8);fig.savefig(OUT/'figures'/f'{name}.png',dpi=150);pdf.savefig(fig);plt.close(fig);pages.append((name,caption))
  def table(title,columns,rows,name,caption='',size=9):
   fig,ax=plt.subplots(figsize=(14,8.5));ax.axis('off');ax.set_title(title,fontsize=19,pad=22);t=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.8)
   for (i,j),cell in t.get_celld().items():cell.set_edgecolor('#cfdbdf');cell.set_facecolor('#dcebee' if i==0 else '#f5f7f8' if i%2 else 'white')
   fig.subplots_adjust(left=.03,right=.97,bottom=.12,top=.88);save(fig,name,caption)
  fig=plt.figure(figsize=(14,8.5));fig.text(.05,.92,'Milestone 2A: current Berkeley road-kernel snapshots',fontsize=22,weight='bold');y=.82
  for text in findings:
   lines=textwrap.fill(text,128);fig.text(.05,y,lines,fontsize=11);y-=.035+.023*len(lines.splitlines())
  save(fig,'01-conclusions','No leader indexing, compaction or memory-layout optimization has been implemented in this experiment.')
  table('Unprofiled timings: three independent repeats, 20 identical steps each',['Case','Active / total','Stopped %','Median µs','Repeat min–max µs','µs / road vehicle'],[[r['name'],f"{r['active']} / {r['total']}",round(100*r['stopped_fraction']),f"{r['time_us']:.2f}",f"{r['min_us']:.2f}–{r['max_us']:.2f}",f"{r['us_per_road_active']:.3f}"] for r in data],'02-timings','Each repeat has three warmups. Input remains immutable; output is never fed back. Allocation, host copies and counter instrumentation are outside reported timings.',8)
  groups=[('Movement',['motion_0','motion_50','motion_90','motion_100']),('Active population',['population_256','population_512','population_1024','motion_50']),('Inactive padding',['motion_50','padding_pending_4096','padding_pending_8192','padding_completed_4096','padding_completed_8192']),('Storage order',['padding_pending_4096','order_shuffle','order_lane','order_interleave','order_state'])]
  for i,(label,names) in enumerate(groups):
   values=[lookup[n] for n in names];columns=['Metric']+[n.replace('_','\n',1) for n in names]
   table(label+' — Nsight GPU and warp metrics',columns,[[LABELS[k]]+[f'{r[METRICS[k]]:.3f}' if r[METRICS[k]] is not None else 'Unavailable' for r in values] for k in ORDER],f'03-{i}-gpu','Separate uninstrumented Nsight run; one measured launch. Fixed clocks; cache flushing disabled to match repeated-state warmup. Counter samples have no confidence interval.',8)
   table(label+' — instruction work and memory traffic',columns,[[label]+[f'{r[k]:,.0f}' if r[k] is not None else 'Unavailable' for r in values] for label,k in zip(EXTRALABEL,EXTRA)],f'04-{i}-memory','Local-allocation launch metric unavailable. Compiler reports advance: 40 registers, 432-byte stack frame, zero spill loads/stores. Local-sector counters show actual local-memory traffic; stack use is not synonymous with register spills.',8)
  table('Explicit search counts from a separate instrumented build',['Case','Road updates','Receiving candidates','Leader candidates','Connector candidates','Stop-line candidates'],[[r['name']]+[f'{r[k]:,}' for k in ('road_updates','receiving_candidates','leader_candidates','connector_candidates','stopline_candidates')] for r in data],'05-counts','Exact per-launch device counters; instrumentation changes execution and is never used for timing or Nsight conclusions. No junction request kernel is executed.',8)
  for name in ('motion_50','padding_pending_8192'):
   rs=[r for r in source if r['case']==name][:6]
   table('Source-correlated long-scoreboard samples: '+name,['CUDA line','Source context','Samples','Share of mapped samples'],[[r['line'],textwrap.fill(r['code'][:130],48),f"{r['samples']:,.0f}",f"{r['share_percent']:.1f}%"] for r in rs],'06-source-'+name,'Production source: src/simulator/junction_simulator.cu. Shares sum over CUDA-line rows only; SASS child rows are not double-counted. Raw CUDA/SASS exports and Nsight captures are retained.',7)
  fig,axes=plt.subplots(1,3,figsize=(14,5))
  axes[0].plot([0,50,90,100],[r['time_us'] for r in motion],'o-');axes[0].set_xlabel('Stopped fraction (%)');axes[0].set_ylabel('GPU µs / step')
  for state in ('pending','completed'):
   rs=[base,lookup[f'padding_{state}_4096'],lookup[f'padding_{state}_8192']];axes[1].plot([r['total'] for r in rs],[r['time_us'] for r in rs],'o-',label=state)
  axes[1].set_xlabel('Total records (2,048 road-active fixed)');axes[1].legend();axes[1].set_ylabel('GPU µs / step')
  rs=[lookup[n] for n in groups[-1][1]];axes[2].bar(range(len(rs)),[r['time_us'] for r in rs]);axes[2].set_xticks(range(len(rs)),['original','shuffle','lane','interleave','state'],rotation=25);axes[2].set_ylabel('GPU µs / step')
  fig.tight_layout(rect=(0,.10,1,1));save(fig,'07-controlled-effects','Geometry, physical state and active population are held fixed within the padding/order comparisons. Speed changes are isolated in the movement comparison.')
  fig=plt.figure(figsize=(14,8.5));fig.text(.05,.91,'Scope, reproducibility and interpretation',fontsize=24,weight='bold')
  notes=[f"Berkeley source: {cfg['roads']:,} roads; {cfg['available_slots']:,} valid lane-0 placement slots. Active subsets are nested and seeded.","20 m placement spacing; moving speed 5 m/s; stopped speed 0; dt=0.25 s; all movements permitted. Positions remain away from stop lines.","One connected outgoing road per fixture route. Connector geometry and conflict arbitration are not exercised; these are constructed performance states.","The unchanged production advance kernel is included directly in the benchmark translation unit, compiled -O3 -lineinfo -arch=sm_87.","128 threads/block. Canonical IDs are external to storage indices; results are compared in canonical order. Admission-priority permutation is not tested here.","Fixed clocks within the existing power mode; original clock settings restored after profiling. Full before/fixed/after readings are retained.","The initial DVFS pass is preserved under exploratory_dvfs and excluded from the reported comparisons.","Normal and instrumented executables are separate. CUDA source, build logs, binary hashes, inputs and profiler commands are preserved.","No claim is made about calibrated trajectories, full-network throughput, intersection request cost, or optimized implementation performance.","Next: independently benchmark leader indexing, including construction/maintenance, against these same fixtures and full-run correctness tests."]
  y=.80
  for text in notes:
   wrapped=textwrap.fill(text,132);fig.text(.05,y,wrapped,fontsize=11);y-=.031+.023*len(wrapped.splitlines())
  save(fig,'08-method','Milestone 1 remains unchanged. This report completes the controlled current-implementation diagnostic phase, not the optimization phase of Milestone 2.')
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Milestone 2A snapshots</title><style>body{max-width:1300px;margin:30px auto;font:18px system-ui}img{width:100%}</style><h1>Milestone 2A — controlled Berkeley snapshots</h1><p><a href="milestone2-snapshots.pdf">PDF report</a> · <a href="metrics.csv">All metrics CSV</a> · <a href="conclusions.md">Conclusions</a></p>'+''.join(f'<img src="figures/{n}.png"><p>{c}</p>' for n,c in pages))
 print(OUT)
if __name__=='__main__':main()
