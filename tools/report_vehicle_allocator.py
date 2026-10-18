#!/usr/bin/env python3
"""Report preregistered vehicle assignment test; no tuning from evaluation data."""
import csv,json,hashlib,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1]; A=R/'artifacts/current/vehicle-allocation-oracle-20261018'; O=R/'reports/vehicle-allocation-oracle-20261018'
NAMES=['Direct','Identity','Geographic','Vehicle count','State grouped','Oracle grouped','Oracle balanced']
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
    for k in ['pre_ms','setup_ms','motion_ms','junction_ms','wall_ms','motion_ms_ratio_count','wall_ms_ratio_count','wall_ms_ratio_direct']:row[k]=median([r[k] for r in rr])
    summary.append(row)
 save('policy_summary',summary)
 gate=[]
 for p in [1,2,4,5,6]:
  rr=[r for r in rows if r['split']=='test' and r['charged']==1 and r['policy']==p]
  batches=[median([r['wall_ms_ratio_count'] for r in rr if r['batch']==b]) for b in [1,2,3]]
  keys=sorted({(r['case'],r['time']) for r in rr}); ratios=[median([r['wall_ms_ratio_count'] for r in rr if (r['case'],r['time'])==key]) for key in keys]
  wins=sum(x<1 for x in ratios)/len(ratios)
  gate.append(dict(policy=p,name=NAMES[p],batch_ratios=batches,win_fraction=wins,windows=len(keys),exact=True,passed=all(x<=.95 for x in batches) and wins>=.75))
 (O/'gate.json').write_text(json.dumps(gate,indent=2))
 bysize=[]
 for n in [2048,8192,16384]:
  for ch in [0,1]:
   for p in range(7):
    rr=[r for r in rows if r['split']=='test' and r['requested']==n and r['charged']==ch and r['policy']==p]
    bysize.append(dict(requested=n,charged=ch,policy=p,name=NAMES[p],motion_ms=median([r['motion_ms'] for r in rr]),wall_ms=median([r['wall_ms'] for r in rr]),wall_ratio_count=median([r['wall_ms_ratio_count'] for r in rr]),motion_ratio_count=median([r['motion_ms_ratio_count'] for r in rr])))
 save('demand_summary',bysize)
 patterns=[]
 for pattern in ['distributed','concentrated','mixed','bursty']:
  for ch in [0,1]:
   for p in range(7):
    rr=[r for r in rows if r['split']=='test' and r['pattern']==pattern and r['charged']==ch and r['policy']==p]
    patterns.append(dict(pattern=pattern,charged=ch,name=NAMES[p],wall_ratio_count=median([r['wall_ms_ratio_count'] for r in rr]),motion_ratio_count=median([r['motion_ms_ratio_count'] for r in rr])))
 save('pattern_summary',patterns)
 passed=[g['name'] for g in gate if g['passed']]
 test={(r['charged'],r['policy']):r for r in summary if r['split']=='test'}
 notes=[
  f"Main finding: keep the direct implementation. With supplied schedules, state grouping reduces complete-step time by {(1-test[0,4]['wall_ms_ratio_count'])*100:.1f}% versus count striping, yet is {(test[0,4]['wall_ms_ratio_direct']-1)*100:.1f}% slower than direct. Count striping itself is {(test[0,3]['wall_ms_ratio_direct']-1)*100:.1f}% slower than direct. Identity indirection changes direct step time by only {(test[0,1]['wall_ms_ratio_direct']-1)*100:.2f}% at the median; task indirection alone does not explain the assignment regressions.",
  f"Supplied oracle grouping and balancing improve count-relative step time by only {(1-test[0,5]['wall_ms_ratio_count'])*100:.2f}% and {(1-test[0,6]['wall_ms_ratio_count'])*100:.2f}%. Charged oracle grouping costs {test[1,5]['wall_ms_ratio_count']:.2f}x count and {test[1,5]['wall_ms_ratio_direct']:.2f}x direct. The evidence does not support training a predictor for these oracle schedules. A locality-preserving assignment mechanism or amortized construction would be a new hypothesis requiring a fresh gate.",
  'Preregistered charged gate: '+(', '.join(passed)+' passed.' if passed else 'No candidate passed.')+' Each candidate needs at least 5% median complete-step improvement in all three batches, wins on at least 75% of 60 held-out snapshots, and exact results.',
  'Scope: 24 frozen Berkeley demand runs, 96 matched snapshots at 60/180/300/450 s; three process batches and ten randomized timing repetitions per policy/cost mode. All 40,320 recorded evaluations passed full vehicle-byte and semantic reservation-summary equivalence; additional motion comparisons and warmups also passed. This is one-step replay, not a new 600-second trajectory run.',
  'All policies run identical vehicle calculations and memory structures. Only stable-ID assignment to threads/blocks changes. The direct original kernel measures indirection overhead; the count policy balances road-active vehicle counts across fixed 128-thread blocks. Geographic and state policies group similar work. Every vehicle record remains present, including pending and completed records.',
  'Oracle candidates rearrange 32-ID tiles from the count schedule. Five instrumented calibration launches measure contextual warp elapsed cycles before independent uninstrumented evaluation. Grouping sorts expensive tiles together; balancing distributes estimated tile cost. These candidates are not optimal upper bounds: regrouping changes locality and observed costs, and tiles preserve their internal thread grouping.',
  'Supplied schedules exclude fitting, list construction and transfer. Charged schedules include state download, CPU assignment, upload and synchronization; oracle schedules additionally include five fitting launches every step. Persistent allocation and correctness/output I/O are excluded for every policy. A reusable, GPU-built or predicted schedule might cost less; this experiment does not measure those implementations.',
  'The complete computational step includes shared leader sorting, motion-context construction and junction processing, which this allocator cannot redistribute. Stage event intervals include host launch gaps; charged setup includes host construction gaps. Wall time is the gate metric. Medians of per-snapshot ratios are not ratios of aggregate medians, and independently medianed stage times need not sum exactly.',
  'No new hardware counters were collected here. Timing differences alone do not establish occupancy, memory-stall or warp-divergence mechanisms. Test demand is held out, but all inputs use the same Berkeley topology and simulator. These results do not establish transfer to another network or simulator.',
  'The identity policy is a control, not a state-aware allocator. If it passes the count-relative gate while the direct implementation is faster, that shows a weakness of count striping rather than evidence for predictive allocation. A policy should improve the deployed direct implementation before being recommended.'
 ]
 (O/'conclusions.md').write_text('\n\n'.join(notes));(O/'protocol.md').write_text((A/'protocol.md').read_text())
 with PdfPages(O/'vehicle-allocation-oracle.pdf') as pdf:
  for start in range(0,len(notes),4):
   fig=plt.figure(figsize=(12,9));fig.text(.05,.95,'Vehicle-work allocation: independent oracle experiment',fontsize=17,va='top');fig.text(.05,.88,'\n\n'.join(textwrap.fill(s,112) for s in notes[start:start+4]),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  for ch in [0,1]:
   ss=[r for r in summary if r['split']=='test' and r['charged']==ch]
   fig,ax=plt.subplots(figsize=(14,6));ax.axis('off');ax.set_title(('Charged' if ch else 'Supplied schedule')+' — held-out snapshots; median per-window measurements')
   tab=ax.table(cellText=[[r['name'],f"{r['motion_ms']*1000:.1f}",f"{r['setup_ms']*1000:.1f}",f"{r['wall_ms']*1000:.1f}",f"{r['motion_ms_ratio_count']:.3f}",f"{r['wall_ms_ratio_count']:.3f}",f"{r['wall_ms_ratio_direct']:.3f}"] for r in ss],colLabels=['Policy','Motion µs','Setup µs','Step wall µs','Motion/count','Step/count','Step/direct'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'table_{ch}.png',dpi=160);plt.close(fig)
  for ch in [0,1]:
   ss=[r for r in summary if r['split']=='test' and r['charged']==ch]
   fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title(('Charged' if ch else 'Supplied schedule')+' stage breakdown — median milliseconds')
   tab=ax.table(cellText=[[r['name'],*[f'{r[k]:.3f}' for k in ['pre_ms','setup_ms','motion_ms','junction_ms','wall_ms']]] for r in ss],colLabels=['Policy','Shared indexes','Setup','Vehicle update','Junction pipeline','Step wall'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2.2);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(14,6))
  for ch,ax in enumerate(axs):
   for p in range(1,7):
    rr=[r for r in bysize if r['charged']==ch and r['policy']==p];ax.plot([r['requested'] for r in rr],[r['wall_ratio_count'] for r in rr],marker='o',label=NAMES[p])
   ax.axhline(1,color='black',lw=1);ax.set_title('Charged' if ch else 'Supplied schedule');ax.set_xlabel('Requested records');ax.set_ylabel('Complete-step wall / vehicle-count policy');ax.set_xscale('log',base=2);ax.legend(fontsize=8)
  fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'step_ratios.png',dpi=160);plt.close(fig)
  fig,ax=plt.subplots(figsize=(13,6));ax.axis('off');ax.set_title('Preregistered gate: charged complete step vs count assignment')
  tab=ax.table(cellText=[[g['name'],*[f'{x:.3f}' for x in g['batch_ratios']],f"{g['win_fraction']*100:.1f}%",'PASS' if g['passed'] else 'FAIL'] for g in gate],colLabels=['Policy','Batch 1 ratio','Batch 2 ratio','Batch 3 ratio','Snapshot wins','Gate'],loc='center');tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.3);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'gate.png',dpi=160);plt.close(fig)
 print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
