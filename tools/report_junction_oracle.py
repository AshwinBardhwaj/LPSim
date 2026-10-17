#!/usr/bin/env python3
import csv,json,statistics as st
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-oracle-20261016';O=R/'reports/berkeley-oracle-20261016'
def main():
 O.mkdir(exist_ok=True);rows=[]
 for path in A.glob('*/*/timings.csv'):
  if not path.parent.name.isdigit():continue
  raw=list(csv.DictReader(path.open()));assert len(raw)==160 and all(x['physical_match']=='true' for x in raw)
  for charged in [0,1]:
   for policy in range(4):
    rr=[x for x in raw if int(x['charged_setup'])==charged and int(x['policy'])==policy]
    rows.append(dict(case=path.parents[1].name,time=int(path.parent.name),charged=charged,policy=policy,gpu_ms=st.median(float(x['gpu_ms']) for x in rr),wall_ms=st.median(float(x['wall_ms']) for x in rr),requests=int(rr[0]['requests']),junctions=int(rr[0]['junctions'])))
 assert len(rows)==288
 for r in rows:
  baseline=next(x for x in rows if x['case']==r['case'] and x['time']==r['time'] and x['charged']==r['charged'] and x['policy']==0)
  r['gpu_ratio']=r['gpu_ms']/baseline['gpu_ms'];r['wall_ratio']=r['wall_ms']/baseline['wall_ms']
 with (O/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 names=['Current indexed order','Compact active junctions','Oracle request grouping','Oracle grouped + spread']
 summary=[]
 for charged in [0,1]:
  for policy in range(4):
   rr=[r for r in rows if r['charged']==charged and r['policy']==policy];summary.append(dict(charged=charged,policy=names[policy],median_gpu_ratio=st.median(r['gpu_ratio'] for r in rr),median_wall_ratio=st.median(r['wall_ratio'] for r in rr),wins=sum(r['wall_ratio']<1 for r in rr),windows=len(rr)))
 gpu=[]
 for path in (A/'profiles').glob('*/metrics.csv'):
  raw=list(csv.DictReader(path.open()));units=raw[0];order=list(csv.DictReader((path.parent/'timings.csv').open()))[:8];assert len(raw)==9
  for x,m in zip(raw[1:],order):
   gpu.append(dict(case=path.parent.name,policy=int(m['policy']),charged=int(m['charged_setup']),duration_us=float(x['gpu__time_duration.sum'])*{'us':1,'ms':1000,'ns':.001}[units['gpu__time_duration.sum']],threads=float(x['smsp__thread_inst_executed_per_inst_executed.ratio']),long_scoreboard=float(x['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']),occupancy=float(x['sm__warps_active.avg.pct_of_peak_sustained_active'])))
 if gpu:
  with (O/'gpu.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(gpu[0]));w.writeheader();w.writerows(gpu)
 (O/'summary.json').write_text(json.dumps(summary,indent=2))
 notes='''This is an actual-request-count oracle, not a perfect execution-cost oracle or a theoretical upper bound on all schedulers. It knows which junctions have requests and their exact counts for the captured step. It does not forecast state.\n\nAll four policies use identical request membership, per-junction priority and admission code. The test changes only junction-thread order and removal of inactive junctions. Grouping sorts descending request count; spread distributes those warp-sized groups across blocks. No policy changes vehicle storage or order inside a junction.\n\n36 matched snapshots: 12 demand/seed/controller combinations at 60, 180 and 450 seconds. Three warmups and twenty randomized-order timing repeats per configuration. Every measured result is checked against full-scan arbitration, including car state and reservation summaries.\n\nFree schedules exclude schedule construction and request-index building, exposing optimistic kernel headroom. Charged schedules include request-index rebuilding for every policy; scheduled variants additionally download ranges, construct CPU schedules, upload IDs and synchronize. Persistent buffers are allocated beforehand. Wall time is the decision metric for charged tests; CUDA event intervals can include host-induced gaps. State restoration and validation are outside timing.\n\nThese are arbitration-stage measurements, not full-step speedups. They do not establish benefits from other scheduling mechanisms, a GPU-resident scheduler, offline amortized schedules or a network prior. Small changes require repeated independent experiment runs before stronger claims. Prior milestones remain unchanged.'''
 notes+='\n\nOne concentrated mode-2 window has no requests: skipping the kernel helps there, but that is an empty-work shortcut rather than evidence for richer state allocation. The grouped schedule improves active threads per instruction in the distributed profiled window, yet that does not establish an elapsed-time benefit. Nsight samples use cold-cache kernel replay separately from warm repeated timing; counter runs are not benchmark timing.'
 findings='Decision gate: no consistent net benefit demonstrated for these request-oracle scheduling policies. Free schedules are approximately flat/slower at the median; CPU construction/transfers add substantial overhead. This rejects proceeding directly to a prior for this particular scheduling mechanism, not state-aware allocation in general. Test a different mechanism or operation before investing in forecasting. Leader sorting remains outside this experiment.'
 (O/'conclusions.md').write_text(findings+'\n')
 (O/'README.md').write_text(notes)
 with PdfPages(O/'oracle-scheduling.pdf') as pdf:
  import textwrap
  fig=plt.figure(figsize=(12,10));fig.text(.05,.95,'Oracle junction scheduling decision gate',fontsize=18,va='top');fig.text(.05,.89,'\n\n'.join(textwrap.fill(p,105) for p in notes.split('\n\n')),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  for charged in [0,1]:
   rr=[r for r in summary if r['charged']==charged];fig,ax=plt.subplots(figsize=(12,5));ax.axis('off');ax.set_title('Setup included' if charged else 'Schedule supplied for free');table=ax.table(cellText=[[r['policy'],f"{r['median_gpu_ratio']:.3f}",f"{r['median_wall_ratio']:.3f}",f"{r['wins']}/36"] for r in rr],colLabels=['Policy','GPU ratio','Wall ratio','Wall-time wins'],loc='center');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'table_{charged}.png',dpi=150);plt.close(fig)
  if gpu:
   fig,ax=plt.subplots(figsize=(12,6));ax.axis('off');ax.set_title('Separate Nsight samples: free schedules at 180 seconds');rr=[r for r in gpu if r['charged']==0];table=ax.table(cellText=[[r['case'].split('_')[0],names[r['policy']],f"{r['duration_us']:.2f}",f"{r['threads']:.2f}",f"{r['long_scoreboard']:.2f}"] for r in rr],colLabels=['Demand','Policy','Duration us','Threads / instruction','Long SB %'],loc='center');table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.8);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'gpu.png',dpi=150);plt.close(fig)
 (O/'index.html').write_text('<h1>Oracle junction scheduling</h1><p>'+findings+'</p><a href="oracle-scheduling.pdf">PDF report</a> | <a href="results.csv">All results</a> | <a href="README.md">Methods</a><br><img width="100%" src="table_0.png"><img width="100%" src="table_1.png">')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
