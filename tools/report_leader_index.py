#!/usr/bin/env python3
"""Render the isolated leader-index experiment without modifying frozen reports."""
import csv,json,statistics,hashlib,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from leader_index_experiment import ROOT,ART
OUT=ROOT/'reports/milestone2-leader-index-20261014'
OUT.mkdir(parents=True,exist_ok=True)
records=json.loads((ART/'timings.json').read_text());rows=[]
for name in dict.fromkeys(r['case'] for r in records):
 b=statistics.median(r['median_ms'] for r in records if r['case']==name and r['variant']=='baseline')
 i=statistics.median(r['median_ms'] for r in records if r['case']==name and r['variant']=='indexed')
 index=[];motion=[]
 for f in sorted((ART/'runs'/name/'indexed').glob('*.costs.csv')):
  samples=list(csv.DictReader(f.open()))
  index.append(statistics.median(float(r['index_ms']) for r in samples))
  motion.append(statistics.median(float(r['motion_ms']) for r in samples))
 rows.append(dict(case=name,baseline_ms=b,indexed_total_ms=i,index_build_ms=statistics.median(index),indexed_motion_ms=statistics.median(motion),speedup=b/i))
with (OUT/'timings.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
metric_keys=[('Motion kernel (ms)','gpu__time_duration.sum'),('SM throughput (%)','sm__throughput.avg.pct_of_peak_sustained_elapsed'),('Occupancy (%)','sm__warps_active.avg.pct_of_peak_sustained_active'),('Eligible warps/cycle','smsp__warps_eligible.avg.per_cycle_active'),('Threads/instruction','smsp__thread_inst_executed_per_inst_executed.ratio'),('Issue active (%)','smsp__issue_active.avg.pct_of_peak_sustained_active'),('Long scoreboard (%)','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),('Wait stall (%)','smsp__warp_issue_stalled_wait_per_warp_active.pct'),('L1/TEX throughput (%)','l1tex__throughput.avg.pct_of_peak_sustained_elapsed'),('L2 throughput (%)','lts__throughput.avg.pct_of_peak_sustained_elapsed'),('Instructions','smsp__inst_executed.sum'),('Global load sectors','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum'),('Local load sectors','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum'),('Registers/thread','launch__registers_per_thread')]
profiles={}
for name in ('motion_50','padding_pending_8192','order_shuffle'):
 for variant in ('baseline','indexed'):
  with (ART/'profiles'/name/variant/'metrics.csv').open() as f:raw=list(csv.DictReader(f))
  profiles[name,variant]=raw[-1]
table=[]
for label,key in metric_keys:
 table.append([label]+[' / '.join(f'{float(profiles[name,v][key].replace(",","")):.3f}' for v in ('baseline','indexed')) for name in ('motion_50','padding_pending_8192','order_shuffle')])
with (OUT/'gpu-metrics.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['metric','2048 active: baseline / indexed','8192 padded: baseline / indexed','4096 shuffled: baseline / indexed']);w.writerows(table)
notes='''Leader indexing: isolated Berkeley road-kernel experiment

All 90 timed outputs and six profiled outputs match frozen Milestone 2A
physical outputs exactly. Three independent repeats, 20 samples each,
three warmups. Fixed clocks; restored afterward. Input is immutable.

Index initialization, sorting, linking and associated device-timeline gaps
are included in total CUDA-event time. No incremental-update shortcut.
Vehicle storage/launch order and other two full-array scans are unchanged.
The timings are warm-cache snapshot steps, not full simulation throughput.

At 2048 active vehicles the full rebuild pays off; at 256 and 512 it does
not. This is evidence against universally enabling this implementation.
The comparator-based sort is an initial implementation, not an optimal
index builder. A radix/counting alternative is a separate experiment.

At 2048 active / 50% stopped, total step time falls 4.244 to 3.490 ms.
Motion instructions fall 5.615M to 3.493M and global load sectors fall
1.151M to 0.755M. Occupancy stays near 16.8%; long-scoreboard stalls stay
near 73%. Lower issue activity accompanies less total work and a faster
step. Percent utilization and stall fractions are not runtime objectives.

Hardware counters describe the motion kernel only, after index construction;
they do not describe aggregate indexing + motion efficiency. Building an
index also changes cache state. Do not treat reduced scan count as the sole
possible cause of differences in counters. Rebuild cost is in timing table.

This index requires distinct valid positions on each road lane, checked by
the fixture validator. No connector occupants or junction requests occur.
Production junction behavior and full Berkeley trajectories remain untested
by this experiment. Current production network-size limits still apply.

Next study: docs/BERKELEY_STATE_ALLOCATION_EXPERIMENT.md. Any allocation
hypothesis remains provisional until full-network and held-out validation.
'''
with PdfPages(OUT/'leader-index.pdf') as pdf:
 fig=plt.figure(figsize=(11.7,16.5));fig.text(.06,.94,notes,va='top',fontsize=12,linespacing=1.45);pdf.savefig(fig);plt.close(fig)
 for title,columns,data in [('Step timings: rebuild included',['Case','Baseline ms','Index build ms','Indexed motion ms','Indexed total ms','Speedup'],[[r['case']]+[f'{r[k]:.3f}' for k in ('baseline_ms','index_build_ms','indexed_motion_ms','indexed_total_ms','speedup')] for r in rows]),('Nsight motion kernel: baseline / indexed',['Metric','2048 active','8192 padded','4096 shuffled'],table)]:
  fig,ax=plt.subplots(figsize=(16,9));ax.axis('off');ax.set_title(title,pad=20)
  t=ax.table(cellText=data,colLabels=columns,loc='center',cellLoc='center');t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
  for (r,c),cell in t.get_celld().items():
   if r==0:cell.set_facecolor('#dcebee')
  fig.tight_layout();pdf.savefig(fig);fig.savefig(OUT/('timings.png' if len(columns)==6 else 'gpu-metrics.png'),dpi=150);plt.close(fig)
text=notes+'\n| Case | Baseline ms | Indexed total ms | Speedup |\n|---|---:|---:|---:|\n'+''.join(f"| {r['case']} | {r['baseline_ms']:.3f} | {r['indexed_total_ms']:.3f} | {r['speedup']:.3f} |\n" for r in rows)
(OUT/'README.md').write_text(text)
# Retain the exact source inputs alongside generated variants and commands.
sources=ART/'sources';sources.mkdir(exist_ok=True)
for p in ('tools/leader_index.cuh','tools/leader_index_experiment.py','tools/snapshot_benchmark.cu','tools/report_leader_index.py','src/simulator/junction_simulator.cu','src/simulator/junction_simulator.h','docs/BERKELEY_STATE_ALLOCATION_EXPERIMENT.md'):
 dest=sources/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
manifest={str(p.relative_to(ART)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ART.rglob('*') if p.is_file() and p.name!='checksums.json'}
(ART/'checksums.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT/'leader-index.pdf')
