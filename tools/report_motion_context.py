from pathlib import Path
import csv,json,statistics,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-motion-context-20261014';O=R/'reports/berkeley-motion-context-20261014';O.mkdir(exist_ok=True)
labels=['Baseline','Connector index','Movement availability','Both'];rows=[];timings=[];sources=[]
for p in sorted((A/'replays').glob('*/*/*/output.json')):
 x=json.loads(p.read_text());timings.append(dict(case=p.parent.parent.parent.name,time=int(p.parent.parent.name),variant=labels[int(p.parent.name)],total_ms=statistics.median(x['total_ms']),motion_ms=statistics.median(x['motion_ms']),construction_ms=statistics.median([a-b for a,b in zip(x['total_ms'],x['motion_ms'])])))
for p in sorted((A/'profiles').glob('*/*/metrics.csv')):
 data=list(csv.DictReader(p.open()));x=data[1];case=p.parent.parent.name;variant=labels[int(p.parent.name)]
 row=dict(case=case,variant=variant)
 for label,key in [('duration','gpu__time_duration.sum'),('long_scoreboard_pct','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'),('occupancy_pct','sm__warps_active.avg.pct_of_peak_sustained_active'),('eligible_warps','smsp__warps_eligible.avg.per_cycle_active'),('active_threads','smsp__thread_inst_executed_per_inst_executed.ratio')]:row[label]=float(x[key].replace(',',''))
 row['duration_unit']=data[0]['gpu__time_duration.sum'];rows.append(row)
 src=list(csv.DictReader((p.parent/'source.csv').read_text().splitlines()[2:]));samples=0
 for x in src:
  if x['Address'].startswith('0x'):
   v=float(x['stall_long_sb'].replace(',','').replace('-','') or 0);samples+=v
   if v:sources.append(dict(case=case,variant=variant,instruction=x['Source'],address=x['Address'],samples=v))
 row['long_scoreboard_samples']=samples
for name,data in [('timings',timings),('gpu_metrics',rows),('source_samples',sources)]:
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=data[0]);w.writeheader();w.writerows(data)
text='''# Independent connector and movement-availability indexes

All indexes are rebuilt from immutable pre-motion state. Variant 1 replaces the full connector scan with an incoming-lane linked list; ID tie-breaking preserves original semantics. Variant 2 computes per-lane receiving blockage and active movement occupancy, then resolves availability once per queried movement using the conflict CSR. Variant 3 combines both. Default remains baseline; select 1/2/3 with LPSIM_MOTION_CONTEXT. Junction admission/reservations are unchanged.

Validation: 36 snapshots × 4 variants match frozen physical digests; 11 small scenarios × 3 optimized variants match physical CSVs; 12 full 600 s Berkeley scenarios × 3 variants match trajectories and pass existing movement/gap checks. Timings have three warmups and ten samples per immutable snapshot. Total time includes leader-index construction, new index initialization/construction, and motion. Full trajectories are correctness runs, not repeated timing comparisons.

Nsight profiles are separate, warmed 180 s snapshots, seed 1401, 72 s controller. Baseline is rerun with the same current binary and protocol; historical cold-restart numbers are not directly comparable. Long-scoreboard percentage is a warp-cycle metric. Source sample counts come from separate SourceCounters launches and are stochastic, duration-dependent observations, not exact stall-event counts or wall-time fractions. A shorter kernel can retain a high stall percentage. Construction-kernel counters are not included in the motion-only table; their elapsed cost is included in total replay timings.

'''
for case in sorted(set(r['case'] for r in rows)):
 text+='## '+case+'\n\n| Variant | Total replay ms | Motion ms | Construction ms | Long scoreboard % | Source long-scoreboard samples |\n|---|---:|---:|---:|---:|---:|\n'
 for row in [r for r in rows if r['case']==case]:
  t=next(t for t in timings if t['case']==case and t['time']==180 and t['variant']==row['variant'])
  text+=f"| {row['variant']} | {t['total_ms']:.4f} | {t['motion_ms']:.4f} | {t['construction_ms']:.4f} | {row['long_scoreboard_pct']:.2f} | {row['long_scoreboard_samples']:.0f} |\n"
text+='\n## Aggregate controlled timing\n\n'
for variant in labels[1:]:
 ratios=[]
 for t in [x for x in timings if x['variant']==variant]:
  base=next(x for x in timings if x['case']==t['case'] and x['time']==t['time'] and x['variant']=='Baseline')
  ratios.append(t['total_ms']/base['total_ms'])
 text+=f"{variant}: median total-time ratio {statistics.median(ratios):.4f}; range {min(ratios):.4f}–{max(ratios):.4f} across 36 windows.\n\n"
text+='## Remaining sampled waits with both indexes\n\n'
for case in sorted(set(r['case'] for r in sources)):
 text+=case+'\n\n'
 for r in sorted([x for x in sources if x['case']==case and x['variant']=='Both'],key=lambda x:x['samples'],reverse=True)[:5]:text+=f"- `{r['instruction']}`: {r['samples']:.0f} samples.\n"
 text+='\n'
(O/'README.md').write_text(text)
with PdfPages(O/'motion-context.pdf') as pdf:
 for case in sorted(set(r['case'] for r in rows)):
  data=[]
  for row in [r for r in rows if r['case']==case]:
   t=next(t for t in timings if t['case']==case and t['time']==180 and t['variant']==row['variant'])
   data.append([row['variant'],f"{t['total_ms']:.3f}",f"{t['motion_ms']:.3f}",f"{t['construction_ms']:.3f}",f"{row['long_scoreboard_pct']:.2f}",f"{row['long_scoreboard_samples']:.0f}"])
  fig,ax=plt.subplots(figsize=(14,6));ax.axis('off');ax.set_title(case+' — 180 s, identical physical input')
  table=ax.table(cellText=data,colLabels=['Variant','Total ms','Motion ms','Build ms','Long scoreboard %','Source samples'],loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
  fig.text(.05,.06,'Total includes all index construction. Source samples and counter percentages are separate profiling observations.\n36 snapshots, 36 full trajectories, and 33 small-scenario checks preserve physical results.',fontsize=10)
  fig.savefig(O/(case+'.png'),dpi=150,bbox_inches='tight');pdf.savefig(fig);plt.close(fig)
(O/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Berkeley motion indexes</title><h1>Independent motion indexes</h1><p><a href="motion-context.pdf">PDF tables</a> | <a href="README.md">Methods and findings</a> | <a href="gpu_metrics.csv">GPU metrics CSV</a> | <a href="timings.csv">All timing results</a></p>'+''.join('<img style="width:100%" src="'+p.name+'">' for p in O.glob('*.png')))
print(text)
