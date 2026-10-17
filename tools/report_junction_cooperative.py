#!/usr/bin/env python3
import csv,json,statistics as st,textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-cooperative-20261016';O=R/'reports/berkeley-cooperative-20261016'
def main():
 O.mkdir(exist_ok=True);rows=[];thresholds=[0,1,2,4,8,16,32]
 for p in (A/'runs').glob('*/*/*/timings.csv'):
  raw=list(csv.DictReader(p.open()));assert len(raw)==280 and all(x['physical_match']=='true' for x in raw)
  f=list(csv.DictReader(Path(str(p)+'.junctions.csv').open()));maxreq=max([int(x['requests']) for x in f] or [0]);maxgrant=max([int(x['admitted']) for x in f] or [0])
  for charged in [0,1]:
   for th in thresholds:
    rr=[x for x in raw if int(x['policy'])==th and int(x['charged_setup'])==charged]
    rows.append(dict(batch=int(p.parents[2].name),case=p.parents[1].name,time=int(p.parent.name),charged=charged,threshold=th,gpu_ms=st.median(float(x['gpu_ms']) for x in rr),wall_ms=st.median(float(x['wall_ms']) for x in rr),requests=int(rr[0]['requests']),junctions=int(rr[0]['junctions']),max_requests=maxreq,max_admissions=maxgrant))
 assert len(rows)==1512
 for r in rows:
  b=next(x for x in rows if all(x[k]==r[k] for k in ['batch','case','time','charged']) and x['threshold']==0);r['wall_ratio']=r['wall_ms']/b['wall_ms'];r['gpu_ratio']=r['gpu_ms']/b['gpu_ms'];r['train']=r['case'].startswith('distributed') and 'seed1401' in r['case']
 with (O/'results.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 summary=[]
 for charged in [0,1]:
  for th in thresholds:
   rr=[r for r in rows if r['charged']==charged and r['threshold']==th];summary.append(dict(charged=charged,threshold=th,gpu_ratio=st.median(r['gpu_ratio'] for r in rr),wall_ratio=st.median(r['wall_ratio'] for r in rr),train_ratio=st.median(r['wall_ratio'] for r in rr if r['train']),test_ratio=st.median(r['wall_ratio'] for r in rr if not r['train']),wins=sum(r['wall_ratio']<1 for r in rr),measurements=len(rr)))
 train=[r for r in summary if r['charged']==1];chosen=min(train,key=lambda r:r['train_ratio']);best_candidate=min([r for r in train if r['threshold']!=0],key=lambda r:r['train_ratio'])
 decision=dict(selected=chosen,best_nonbaseline=best_candidate,heldout_note='24 unique held-out windows, each measured in three batches; repeated batches are not independent traffic cases')
 (O/'decision.json').write_text(json.dumps(decision,indent=2));(O/'summary.json').write_text(json.dumps(summary,indent=2))
 notes=[
 'Experimental implementation: one warp per junction. Conflict queries against prior grants run in parallel. Priority-dependent admission, gap checks and reservation updates remain ordered and single-writer. All paths use the same indexed requests.',
 'Threshold 0 denotes the original one-thread-per-junction baseline. Threshold 1 uses the cooperative path for every nonempty junction. Thresholds 2–32 run two kernels: the scalar path below the threshold and the cooperative path at/above it. GPU-side range-length checks dispatch work without CPU transfers or task sorting. Both launches and synchronization are included.',
 '36 matched Berkeley snapshots at 60, 180 and 450 seconds; three independent process batches, each with three warmups and 20 randomized timing repeats per configuration. All car-state and semantic reservation arrays are checked against full-scan reference arbitration after every run. Separate exclusive-mode checks use 12 snapshots.',
 'Charged timings include request-index rebuilding; uncharged timings isolate dispatch and arbitration with the index available. Restoration and validation are outside timing. Wall time includes launch and synchronization overhead. These are stage timings, not complete-simulation speedups.',
 'Threshold selection uses distributed demand with seed1401 only (12 windows). Seed1402 and concentrated demand are held out (24 windows). Choose the lowest median normalized charged wall time, allowing the baseline as a valid choice. Actual current request counts are used, not predicted counts: this tests an optimistic decision rule before forecasting.',
 'Only Berkeley is tested; neither unseen-network transfer nor a learned allocator is demonstrated. The cooperative path parallelizes only conflict checks against new grants, so it is not an upper bound on other cooperative algorithms. A physical junction can have many requests but few new grants, limiting the useful parallel work.'
 ]
 observations=[]
 for p in (A/'runs/1').glob('*/*/timings.csv.junctions.csv'):observations.extend(list(csv.DictReader(p.open())))
 from collections import Counter
 distribution=dict(requests=dict(Counter(int(x['requests']) for x in observations)),admissions=dict(Counter(int(x['admitted']) for x in observations)))
 (O/'junction_distribution.json').write_text(json.dumps(distribution,indent=2))
 notes.append(f'Across {len(observations)} active-junction observations, request-list sizes ranged from 1 to {max(int(x["requests"]) for x in observations)}. Admission counts ranged from 0 to {max(int(x["admitted"]) for x in observations)}. Long traffic queues did not create long admission lists in these captured states. Thus the conflict-check cooperative path had very little parallel work; these data do not establish its behavior at much larger request/grant counts.')
 conclusion=f"Training-selected threshold: {chosen['threshold']} (0 means retain baseline). Held-out charged wall ratio: {chosen['test_ratio']:.3f}. Best nonbaseline threshold: {best_candidate['threshold']}; held-out ratio {best_candidate['test_ratio']:.3f}. Ratios above 1 are slower. No learned model was trained."
 (O/'conclusions.md').write_text(conclusion+'\n');(O/'README.md').write_text('\n\n'.join(notes))
 with PdfPages(O/'cooperative-junctions.pdf') as pdf:
  fig=plt.figure(figsize=(12,12));fig.text(.05,.96,'Cooperative junction processing and threshold gate',fontsize=17,va='top');fig.text(.05,.91,'\n\n'.join(textwrap.fill(x,110) for x in [conclusion]+notes),fontsize=10,va='top');pdf.savefig(fig);plt.close(fig)
  for charged in [0,1]:
   rr=[r for r in summary if r['charged']==charged];fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title('Index construction included' if charged else 'Dispatch + arbitration; index prebuilt');t=ax.table(cellText=[[r['threshold'],f"{r['gpu_ratio']:.3f}",f"{r['wall_ratio']:.3f}",f"{r['train_ratio']:.3f}",f"{r['test_ratio']:.3f}",str(r['wins'])+'/108'] for r in rr],colLabels=['Threshold','GPU ratio','Wall ratio','Training wall','Held-out wall','Wins'],loc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'table_{charged}.png',dpi=140);plt.close(fig)
  profiles=[]
  for p in (A/'profiles').glob('*/metrics.csv'):
   raw=list(csv.DictReader(p.open()));units=raw[0];order=list(csv.DictReader((p.parent/'timings.csv').open()))[:14];assert len(raw)==15
   for x,m in zip(raw[1:],order):
    if int(m['policy']) not in [0,1]:continue
    profiles.append(dict(case=p.parent.name,threshold=int(m['policy']),charged=int(m['charged_setup']),duration_us=float(x['gpu__time_duration.sum'])*{'us':1,'ms':1000,'ns':.001}[units['gpu__time_duration.sum']],threads=float(x['smsp__thread_inst_executed_per_inst_executed.ratio']),long_scoreboard=float(x['smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct'])))
  if profiles:
   with (O/'gpu.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
   fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title('Separate cold-cache Nsight samples, 180 s; stage timings above decide performance');rr=[r for r in profiles if r['charged']==0];t=ax.table(cellText=[[r['case'].split('_')[0],r['threshold'],f"{r['duration_us']:.2f}",f"{r['threads']:.2f}",f"{r['long_scoreboard']:.2f}"] for r in rr],colLabels=['Demand','Threshold','Kernel us','Threads / instruction','Long SB %'],loc='center');t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'gpu.png',dpi=140);plt.close(fig)
 (O/'index.html').write_text('<h1>Cooperative junction processing</h1><p>'+conclusion+'</p><a href="cooperative-junctions.pdf">PDF</a> | <a href="results.csv">Measurements</a> | <a href="README.md">Methods</a><img width="100%" src="table_0.png"><img width="100%" src="table_1.png">')
 print(json.dumps(decision,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
