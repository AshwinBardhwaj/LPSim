#!/usr/bin/env python3
import json,csv,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-block-imbalance-20261019';O=R/'reports/berkeley-block-imbalance-20261019';N=R/'data/networks/berkeley_junction_study'
DT=np.dtype(dict(names=['state','k','edges','lanes','pos','speed','departure'],formats=['i4','i4',('i4',32),('i4',32),'f4','f4','f4'],offsets=[0,4,24,152,412,416,408],itemsize=428))
def med(v):return float(np.median(v))
def save(name,rows):
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 assert json.loads((A/'replay_complete.json').read_text())['passed'];O.mkdir(exist_ok=True)
 cfg=json.loads((A/'prepared.json').read_text());roads=json.loads((N/'geometry.json').read_text())['roads'];length=np.array([r['length'] for r in roads]);major=np.array([r['lanes']>=2 or r['speed']/.44704>=34.9 for r in roads]);windows=[];features=[];launches=[];runs=[];traces={}
 for c in cfg['cases']:
  m=json.loads((A/'runs'/c['name']/'metrics.json').read_text());series=list(csv.DictReader((A/'runs'/c['name']/'step_metrics.csv').open()))
  runs.append(dict(case=c['name'],pattern=c['pattern'],requested=c['n'],completed=m['completed'],final_active=m['active'],final_pending=m['pending'],peak_road_active=max(int(r['road_stopped'])+int(r['road_moving']) for r in series),peak_moving=max(int(r['road_moving']) for r in series),major_road_distance_share=c['major_road_distance_share']))
  for t in [60,180,300,450]:
   raw=(A/'runs'/c['name']/f'snapshot_{t}.bin').read_bytes();v=np.frombuffer(raw,dtype=DT,offset=8);assert len(v)==c['n'];states=v['state'];edge=v['edges'][np.arange(len(v)),v['k']];road=states==1;moving=road&(v['speed']>=.5);stopped=road&~moving;pending=states==0;connector=states==2;done=states==3;near=road&(length[edge]-v['pos']<30);nb=c['n']//128
   leaders=np.zeros(len(v),dtype=bool);ids=np.flatnonzero(road)
   if len(ids):
    lane=v['lanes'][np.arange(len(v)),v['k']];keys=edge[ids]*8+lane[ids];order=np.lexsort((v['pos'][ids],keys));si=ids[order];sk=keys[order];leaders[si[np.r_[sk[1:]!=sk[:-1],True]]]=True
   masks=dict(road_active=road,moving=moving,stopped=stopped,connector=connector,pending=pending,completed=done,near_end=near,leaders=leaders,near_end_leaders=leaders&near,major_road_active=road&major[edge])
   counts={k:mask.reshape(nb,128).sum(axis=1) for k,mask in masks.items()};durs={k:[] for k in range(nb)};finishes={k:[] for k in range(nb)};timings={i:[] for i in [0,1]};winlaunch=[]
   for batch in [1,2,3]:
    p=A/'replays'/str(batch)/c['name']/str(t);ts=list(csv.DictReader((p/'timings.csv').open()));assert len(ts)==20 and all(r['exact']=='true' for r in ts)
    for i in [0,1]:timings[i].extend([r for r in ts if int(r['instrument'])==i])
    bs=list(csv.DictReader((p/'timings.csv.blocks.csv').open()));assert len(bs)==nb*10
    for rep in range(10):
     rr=[r for r in bs if int(r['repeat'])==rep];bids=np.array([int(r['block']) for r in rr]);start=np.array([int(r['start_ns']) for r in rr],dtype=np.int64);end=np.array([int(r['end_ns']) for r in rr],dtype=np.int64);sm=np.array([int(r['sm']) for r in rr]);origin=start.min();start=(start-origin)/1e6;end=(end-origin)/1e6;dur=end-start;span=end.max();cut=np.sort(end)[int(np.ceil(.9*nb))-1];smend=np.array([end[sm==s].max() for s in np.unique(sm)]);median=med(dur)
     row=dict(case=c['name'],pattern=c['pattern'],requested=c['n'],time=t,batch=batch,repeat=rep,blocks=nb,sm_count=len(smend),span_ms=span,duration_median_ms=median,duration_p90_ms=float(np.quantile(dur,.9)),duration_max_ms=float(max(dur)),max_over_median=float(max(dur)/median),p90_over_median=float(np.quantile(dur,.9)/median),completion_tail_fraction=float((span-cut)/span),start_spread_fraction=float((start.max()-start.min())/span),sm_finish_spread_fraction=float((max(smend)-min(smend))/span),road_duration_correlation=float(np.corrcoef(counts['road_active'][bids],dur)[0,1]) if np.std(counts['road_active']) and np.std(dur) else None)
     launches.append(row);winlaunch.append(row)
     for bid,d,e in zip(bids,dur,end):durs[bid].append(float(d));finishes[bid].append(float(e/span))
     if batch==1 and rep==5:traces[c['name'],t]=(start,end,sm)
   row=dict(case=c['name'],pattern=c['pattern'],requested=c['n'],time=t,blocks=nb,**{k:int(mask.sum()) for k,mask in masks.items()},future=int(np.count_nonzero(pending&(v['departure']>t))),eligible_pending=int(np.count_nonzero(pending&(v['departure']<=t))))
   for k in ['pre_ms','leader_ms','context_ms','motion_ms','junction_ms','pipeline_ms','wall_ms']:row[k]=med([float(r[k]) for r in timings[0]])
   row['instrument_motion_ratio']=med([float(r['motion_ms']) for r in timings[1]])/row['motion_ms']
   for k in ['span_ms','duration_median_ms','duration_p90_ms','duration_max_ms','max_over_median','p90_over_median','completion_tail_fraction','start_spread_fraction','sm_finish_spread_fraction']:row[k]=med([r[k] for r in winlaunch])
   cc=[r['road_duration_correlation'] for r in winlaunch if r['road_duration_correlation'] is not None];row['road_duration_correlation']=med(cc) if cc else None;windows.append(row)
   for bid in range(nb):features.append(dict(case=c['name'],pattern=c['pattern'],requested=c['n'],time=t,block=bid,**{k:int(counts[k][bid]) for k in counts},duration_median_ms=med(durs[bid]),normalized_finish_median=med(finishes[bid])))
 save('runs',runs);save('windows',windows);save('blocks',features);save('instrumented_launches',launches)
 summary=[]
 for pattern in ['distributed','concentrated','bursty']:
  for n in [2048,8192,16384]:
   rr=[r for r in windows if r['pattern']==pattern and r['requested']==n];row=dict(pattern=pattern,requested=n)
   for k in ['road_active','moving','stopped','pending','pre_ms','leader_ms','context_ms','motion_ms','junction_ms','wall_ms','max_over_median','p90_over_median','completion_tail_fraction','start_spread_fraction','sm_finish_spread_fraction','instrument_motion_ratio']:row[k]=med([r[k] for r in rr])
   summary.append(row)
 save('summary',summary)
 correlations=[]
 for w in windows:
  rr=[r for r in features if r['case']==w['case'] and r['time']==w['time']];y=[r['duration_median_ms'] for r in rr]
  for k in ['block','road_active','moving','stopped','pending','connector','leaders','near_end_leaders','major_road_active']:
   x=[r[k] for r in rr]
   if np.std(x) and np.std(y):correlations.append(dict(case=w['case'],time=w['time'],requested=w['requested'],feature=k,correlation=float(np.corrcoef(x,y)[0,1])))
 save('block_correlations',correlations)
 resources=list(csv.DictReader((A/'replays/1'/cfg['cases'][0]['name']/'60/timings.csv.resources.csv').open()));save('kernel_resources',resources)
 notes=[
  f"Across 36 matched windows, the median block max/median duration ratio is {med([r['max_over_median'] for r in windows]):.2f}; the median p90/median ratio is {med([r['p90_over_median'] for r in windows]):.2f}. The median last-10%-completion tail is {100*med([r['completion_tail_fraction'] for r in windows]):.1f}% of instrumented kernel span, ranging {100*min(r['completion_tail_fraction'] for r in windows):.1f}–{100*max(r['completion_tail_fraction'] for r in windows):.1f}%. These describe heterogeneity and finishing tails, not achievable queue speedups.",
  'Kernel resource check: '+ '; '.join(f"{r['variant']}: {r['registers']} registers/thread, {r['shared_bytes']} shared bytes/block, {r['local_bytes']} local bytes/thread, theoretical maximum {r['max_resident_blocks_per_sm']} resident blocks/SM" for r in resources)+'. Resource changes are an instrumentation limitation; they are not an optimization result.',
 'Scope: nine new Berkeley trajectories, seed 1901, 600 seconds each; 36 matched windows; three independent replay processes and ten repetitions per clean/instrumented mode. All measured vehicle bytes and semantic reservation summaries match original. Full-run movement, gap, conflict and population checks pass. One seed and one network limit generalization.',
 f"Demand weights: 75% uniform route-pool mixture plus 25% normalized lane/speed importance. Major-road proxy: >=2 lanes or >=35 mph. Unweighted route-pool main-road distance share is {cfg['unweighted_route_pool_major_share']*100:.1f}%; sampled planned distance averages {np.mean([r['major_road_distance_share'] for r in runs])*100:.1f}%. This is synthetic demand, not calibrated Berkeley counts. Concentrated patterns also bias destination regions.",
 'Safe task: consecutive stable vehicle IDs, reading immutable pre-motion state and writing disjoint next-state records. Prebuilt leader and motion-context indexes remain shared read-only dependencies. Stopped vehicles remain eligible for processing. Pending/completed record handling cannot simply disappear. Junction admission and reservation priority remain separate, unchanged stages.',
 f"Instrumentation motion/event ratio: median {med([r['instrument_motion_ratio'] for r in windows]):.3f}, range {min(r['instrument_motion_ratio'] for r in windows):.3f}–{max(r['instrument_motion_ratio'] for r in windows):.3f}. Globaltimer timestamps are cross-SM comparable. Entry/end barriers and logging alter execution; block measurements describe the instrumented kernel and are diagnostic, not baseline latency or attainable speedup.",
 'Block duration includes memory waiting, scheduling and barriers, not isolated useful computation. Completion tail = time after 90% of blocks have finished until the last block finishes, divided by first-start to last-finish span. Start spread and final SM finish spread help distinguish late admission/waves from long-running blocks. Neither tail nor duration imbalance proves a dynamic queue will improve runtime.',
 'Clean step timings include shared index construction, vehicle motion and junction stages; host wall includes synchronization. Output/validation I/O is excluded from replay timing. Stage medians need not sum exactly. The nine full-run wall times include diagnostics and are not used as allocator performance claims.',
 'State associations are descriptive. Original IDs are departure-sorted, so pending/future, road population, topology and storage position covary. No state feature is established as causal by these correlations. No compaction or queue scheduler is implemented in this study.'
 ]
 for n in [2048,8192,16384]:
  rr=[r for r in windows if r['requested']==n];notes.append(f"Requested {n}: median last-completion tail {100*med([r['completion_tail_fraction'] for r in rr]):.2f}%, block-start spread {100*med([r['start_spread_fraction'] for r in rr]):.2f}%, and final SM finish spread {100*med([r['sm_finish_spread_fraction'] for r in rr]):.2f}% of kernel span. Median block duration vs road-active count correlation is {med([r['road_duration_correlation'] for r in rr if r['road_duration_correlation'] is not None]):.2f}. A larger road population does not consistently identify the slowest blocks.")
 notes.append('Interpretation: measurable block-duration variation exists, especially at 8192 records, but few-block finishing tails are small and SM completion times are closely aligned. At 16384 records the block-start spread reveals multiple scheduling waves; treating that start delay as task execution cost would be misleading. This study does not demonstrate a large recoverable imbalance for a persistent queue. Shared leader-index construction remains a major complete-step cost outside the proposed motion allocator.')
 notes.append('Leader indexing consumes median '+', '.join(f"{100*med([r['leader_ms']/r['wall_ms'] for r in windows if r['requested']==n]):.1f}% at {n} requested" for n in [2048,8192,16384])+'. This includes initialization/sort/link stages and associated device-timeline gaps; motion redistribution does not remove this shared cost.')
 notes.append('Block ID vs elapsed-duration correlations have medians '+', '.join(f"{med([r['correlation'] for r in correlations if r['requested']==n and r['feature']=='block']):.2f} at {n} requested" for n in [2048,8192,16384])+'. The timeline shows an ordered progression of finishes within resident groups. This is evidence that elapsed block durations include execution-order/resource-sharing effects, not trustworthy labels of independent traffic-task compute cost. It does not identify the specific hardware mechanism.')
 (O/'conclusions.md').write_text('\n\n'.join(notes));(O/'specification.md').write_text((A/'specification.md').read_text())
 with PdfPages(O/'block-imbalance.pdf') as pdf:
  for i in range(0,len(notes),4):
   fig=plt.figure(figsize=(12,9));fig.text(.05,.95,'Berkeley: safe motion tasks and block imbalance',fontsize=17,va='top');fig.text(.05,.88,'\n\n'.join(textwrap.fill(s,110) for s in notes[i:i+4]),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  def table(title,labels,data,name):
   fig,ax=plt.subplots(figsize=(15,8));ax.axis('off');ax.set_title(title);tab=ax.table(cellText=data,colLabels=labels,loc='center');tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/(name+'.png'),dpi=150);plt.close(fig)
  table('Matched-window medians: clean stage time and actual traffic',['Pattern','Requested','Road active','Stopped','Leader ms','Context ms','Motion ms','Junction ms','Step wall ms'],[[r['pattern'],r['requested'],round(r['road_active']),round(r['stopped']),*[f"{r[k]:.3f}" for k in ['leader_ms','context_ms','motion_ms','junction_ms','wall_ms']]] for r in summary],'stages')
  table('Instrumented block imbalance: descriptive, not a predicted speedup',['Pattern','Requested','p90/median','Max/median','Last 10% tail','Start spread','SM finish spread','Instrument/baseline'],[[r['pattern'],r['requested'],f"{r['p90_over_median']:.2f}",f"{r['max_over_median']:.2f}",*[f"{100*r[k]:.1f}%" for k in ['completion_tail_fraction','start_spread_fraction','sm_finish_spread_fraction']],f"{r['instrument_motion_ratio']:.3f}"] for r in summary],'imbalance')
  table('Full-run traffic outcomes at 600 seconds',['Pattern','Requested','Peak road active','Peak moving','Final active','Pending','Completed'],[[r['pattern'],r['requested'],r['peak_road_active'],r['peak_moving'],r['final_active'],r['final_pending'],r['completed']] for r in runs],'populations')
  for c in cfg['cases']:
   start,end,sm=traces[c['name'],300];fig,ax=plt.subplots(figsize=(13,6));ax.hlines(np.arange(len(start)),start,end,colors=plt.cm.tab10(sm%10),lw=3);ax.set_xlabel('Milliseconds from first block start (instrumented)');ax.set_ylabel('Original contiguous block ID');ax.set_title(c['name']+' at 300 s — batch 1, repetition 5');fig.tight_layout();pdf.savefig(fig);fig.savefig(O/(c['name']+'_timeline.png'),dpi=130);plt.close(fig)
  fig,axs=plt.subplots(1,2,figsize=(13,6))
  for n in [2048,8192,16384]:
   rr=[r for r in features if r['requested']==n];axs[0].scatter([r['road_active'] for r in rr],[r['duration_median_ms'] for r in rr],s=5,alpha=.25,label=str(n))
  axs[0].set_xlabel('Road-active records in a 128-record block');axs[0].set_ylabel('Median instrumented block duration (ms)');axs[0].legend();axs[1].scatter([r['start_spread_fraction']*100 for r in windows],[r['completion_tail_fraction']*100 for r in windows]);axs[1].set_xlabel('Block-start spread (% kernel span)');axs[1].set_ylabel('Last 10% completion tail (% span)');fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'state_and_tail.png',dpi=150);plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
