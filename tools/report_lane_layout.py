"""Report physical lane-layout costs and the predeclared decision gate."""
from pathlib import Path
import json,argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,default=root/'artifacts/current/lane-layout-20261022');p.add_argument('--output',type=Path,default=root/'reports/lane-layout-20261022');args=p.parse_args();a=args.artifact;r=args.output;r.mkdir(parents=True,exist_ok=True)
frames=[]
for f in sorted((a/'replays').glob('*.csv')):
 d=pd.read_csv(f);pattern,n,seed,t,batch=f.stem.split('_');d['pattern']=pattern;d['n']=int(n);d['t']=int(t);d['batch']=int(batch);frames.append(d)
d=pd.concat(frames,ignore_index=True);assert len(d)==3240 and d.exact.all();d.to_csv(r/'timings.csv',index=False)
metrics=['index_ms','pack_ms','context_ms','motion_ms','unpack_ms','junction_ms','pipeline_ms','wall_ms']
w=d.groupby(['pattern','n','t','batch','variant'])[metrics].median().reset_index();w.to_csv(r/'windows.csv',index=False)
s=w.groupby(['n','variant'])[metrics].median();s.to_csv(r/'summary.csv')
pair=w.pivot(index=['pattern','n','t','batch'],columns='variant',values='wall_ms');pair['charged_speedup']=pair.current/pair.lane_charged;pair['supplied_speedup']=pair.current/pair.lane_supplied;pair.to_csv(r/'paired_speedups.csv')
# Context counts and conservative layout-persistence evidence from neighboring saved states.
base=root/'artifacts/current/berkeley-block-imbalance-20261019/runs';pop=[]
for pattern,n,t in w[['pattern','n','t']].drop_duplicates().itertuples(index=False,name=None):
 words=np.frombuffer((base/f'{pattern}_{n}_seed1901'/f'snapshot_{t}.bin').read_bytes()[8:],dtype=np.int32).reshape(n,107);state=words[:,0];speed=words[:,104].copy().view(np.float32);road=state==1
 pop.append(dict(pattern=pattern,n=n,t=t,road_active=int(road.sum()),road_stopped=int((road&(speed<.5)).sum()),road_moving=int((road&(speed>=.5)).sum()),pending=int((state==0).sum()),connector=int((state==2).sum()),completed=int((state==3).sum())))
pop=pd.DataFrame(pop);pop['stopped_fraction']=pop.road_stopped/pop.road_active;pop.to_csv(r/'populations.csv',index=False)
state=pair.reset_index().merge(pop,on=['pattern','n','t']);state.to_csv(r/'state_costs.csv',index=False)
# A workload subset must improve in each of three process batches, not just its best repetition.
gate=pair['charged_speedup'].groupby(level=['pattern','n','t']).min();wins=int((gate>=1.05).sum())
lines=['# Physical lane-order layout experiment','',f'Practical gate: {wins}/36 windows achieved >=5% complete-step improvement in all three process batches. Maximum charged speedup in any paired window/batch: {pair.charged_speedup.max():.3f}x. Values below 1 mean slower.','', '| Requested vehicles | Current step ms | Charged lane step ms | Supplied lane step ms | Current / lane motion ms |','|---:|---:|---:|---:|---:|']
for n in [2048,8192,16384]:
 c=s.loc[n,'current'];l=s.loc[n,'lane_charged'];o=s.loc[n,'lane_supplied'];lines.append(f'| {n:,} | {c.wall_ms:.3f} | {l.wall_ms:.3f} | {o.wall_ms:.3f} | {c.motion_ms:.3f} / {l.motion_ms:.3f} |')
lines+=['','This changes physical Car storage for motion, not merely the order of a task list. Gather records in lane/position order, invert the slot map and remap leaders, build connector/context references against the packed snapshot, execute the same motion kernel, then scatter results to stable-ID order for unchanged junction arbitration. All records remain present. Non-road records preserve relative stable-ID order, including connector tie-breaking.','', 'The first whole-record-per-thread copy adapter was expensive. It is retained under record_copy_diagnostic and excluded from the final tables. Final gather/scatter kernels distribute contiguous 32-bit fields across threads. The final matrix uses this improved adapter throughout; no timing samples were selected by speed.', '', 'Charged layout includes every gather, inverse map, leader remap and scatter each step. Supplied layout excludes input gather/remapping but still charges the stable-ID junction boundary scatter and rebuilds the common leader/context indexes. It is an optimistic repeated-snapshot test, not a persistent layout maintained across evolving simulation steps. Both include unchanged junction work and terminal synchronization. Output, host diagnostics and one-time allocation are excluded. Additional persistent adapter memory: two Car arrays plus two integer maps, 864 bytes per requested vehicle (13.5 MiB at 16,384).', '', 'Validation: 3,240 measured replays across 36 windows and three separate process batches; three warmups and ten randomized measurements per variant. Every replay matches canonical-ID post-motion and final Car bytes plus nearest/arrival/tail, busy/reserved/owner/spawn summaries. These are per-step snapshot checks, not new full-trajectory runs. Fixed Jetson 15 W / 612 MHz GPU / 2133 MHz EMC; fast-math matches the frozen simulator. Original clock settings were restored.', '', 'Decision: '+('The gate passed for some subsets; dispatch-only controls and a held-out selector are still required before claiming a useful traffic prior.' if wins else 'The practical gate failed. Do not deploy this adapter or train a prior to choose it on these workloads. The frozen production radix implementation remains unchanged.'), '', 'Interpretation: lane ordering alone leaves the 428-byte array-of-structures record stride and the motion kernel calculations unchanged. The timing table measures whether its locality/grouping benefit repays copying; it does not establish which individual loads dominate. No Nsight counters were collected, so reduced memory stalls or improved occupancy are not claimed. A persistent whole-pipeline layout or hot/cold field split is a different, untested design. These results do not rule it out, but a prior cannot recover costs that the tested layout does not save.', '', f"Stopped road-active fractions ranged from {pop.stopped_fraction.min():.1%} to {pop.stopped_fraction.max():.1%}. {(pop.stopped_fraction>=.5).sum()} of 36 windows had a stopped majority; none achieved a charged-layout win. These fractions describe current snapshots, not a prediction of queue persistence.", '', 'The planned selector would need to beat current storage everywhere, lane storage everywhere and a cheap current-state rule, with bursty demand held out. No learned-policy or forecast benefit has been demonstrated here. See state_costs.csv for stopped/moving populations alongside paired timings; this is descriptive association, not causality.', '', 'Reproduce: compile tools/lane_layout_replay.cu with nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17 into NEW_DIRECTORY/replay. Run tools/experiment_lane_layout.py --output NEW_DIRECTORY, then tools/report_lane_layout.py --artifact NEW_DIRECTORY --output NEW_REPORT_DIRECTORY. Preserve the frozen Berkeley input milestone; do not overwrite archived results.']
(r/'README.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,metric,title in zip(axes,['wall_ms','motion_ms'],['Complete replayed step','Motion kernel only']):
 for j,v in enumerate(['current','lane_charged','lane_supplied']):ax.bar([i+(j-1)*.25 for i in range(3)],[s.loc[(n,v),metric] for n in [2048,8192,16384]],width=.25,label=v)
 ax.set_xticks(range(3),['2,048','8,192','16,384']);ax.set_ylabel('Milliseconds');ax.set_xlabel('Requested vehicles');ax.set_title(title);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(r/'layout-timings.png',dpi=180)
with PdfPages(r/'lane-layout.pdf') as pdf:
 pdf.savefig(fig);plt.close(fig)
 import textwrap
 text='\n\n'.join('\n'.join(textwrap.wrap(x.replace('#','').strip(),105)) for x in lines if x and not x.startswith('|'));rows=text.splitlines()
 for i in range(0,len(rows),46):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.95,'\n'.join(rows[i:i+46]),va='top',fontsize=10);pdf.savefig(fig);plt.close(fig)
print(s.to_string());print('passing windows:',wins,'maximum charged speedup:',pair.charged_speedup.max())
