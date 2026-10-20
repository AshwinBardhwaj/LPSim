"""Summarize paired replay timings without selecting best samples."""
from pathlib import Path
import json,argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--artifact',type=Path,default=root/'artifacts/current/leader-radix-20261019');parser.add_argument('--output',type=Path,default=root/'reports/leader-radix-20261019');args=parser.parse_args();a=args.artifact;r=args.output;r.mkdir(parents=True,exist_ok=True)
frames=[]
for f in sorted((a/'replays').glob('*.csv')):
 d=pd.read_csv(f);pattern,n,seed,t,batch=f.stem.split('_');d['pattern']=pattern;d['n']=int(n);d['t']=int(t);d['batch']=int(batch);d['case']=f'{pattern}_{n}_{seed}';frames.append(d)
d=pd.concat(frames,ignore_index=True);assert len(d)==2160 and d.exact.all()
# Population context is decoded outside timing from the immutable replay inputs.
import numpy as np
pop=[]
for case,t in sorted({(x['case'].iloc[0],int(x['t'].iloc[0])) for x in frames}):
 raw=(root/'artifacts/current/berkeley-block-imbalance-20261019/runs'/case/f'snapshot_{t}.bin').read_bytes()
 count=int(np.frombuffer(raw[:4],dtype=np.int32)[0]);words=np.frombuffer(raw[8:],dtype=np.int32).reshape(count,107);state=words[:,0];speed=words[:,104].copy().view(np.float32)
 pop.append(dict(case=case,time=t,requested=count,pending=int((state==0).sum()),road_active=int((state==1).sum()),road_stopped=int(((state==1)&(speed<.5)).sum()),road_moving=int(((state==1)&(speed>=.5)).sum()),connector=int((state==2).sum()),completed=int((state==3).sum())))
pd.DataFrame(pop).to_csv(r/'populations.csv',index=False)

d['leader_ms']=d.prepare_ms+d.sort_ms+d.link_ms;d.to_csv(r/'timings.csv',index=False)
cols=['prepare_ms','sort_ms','link_ms','leader_ms','context_ms','motion_ms','junction_ms','pipeline_ms','wall_ms']
w=d.groupby(['case','n','t','batch','variant'])[cols].median().reset_index();w.to_csv(r/'windows.csv',index=False)
summary=w.groupby(['n','variant'])[cols].median();summary.to_csv(r/'summary.csv')
p=w.pivot(index=['case','n','t','batch'],columns='variant',values='wall_ms');p['speedup']=p.comparator/p.radix;assert p.speedup.min()>1;p.to_csv(r/'paired_speedups.csv')
trajectories=json.loads((a/'trajectory_equivalence.json').read_text());assert len(trajectories)==9 and all(all(x['checks'].values()) for x in trajectories)
assert all(all(x['metrics'][k]==0 for k in ['gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples']) for x in trajectories)
lines=['# Leader-index reconstruction: stable radix sort','', 'The comparator sort dominated reconstruction time. Materializing lane/position keys once and using a stable radix sort reduces repeated scattered Car reads and reuses persistent scratch. IDs, vehicle storage, launches, and physical calculations remain unchanged.','', '## Paired snapshot results','', '| Requested vehicles | Index before / after (ms) | Complete step before / after (ms) | Step speedup |','|---:|---:|---:|---:|']
for n in [2048,8192,16384]:
 old=summary.loc[(n,'comparator')];new=summary.loc[(n,'radix')]
 lines.append(f'| {n:,} | {old.leader_ms:.3f} / {new.leader_ms:.3f} | {old.wall_ms:.3f} / {new.wall_ms:.3f} | {old.wall_ms/new.wall_ms:.2f}x |')
lines+=['', 'Medians of per-window/per-process medians; all 36 matched windows, three process batches, three warmups and ten measured repetitions per method, randomized paired order. Timings include key generation, sorting, linking, unchanged context/motion/intersection stages and terminal synchronization. Jetson 15 W mode, fixed 612 MHz GPU and 2133 MHz EMC; 128-thread motion/key/link blocks. No instrumentation counters or GPU profiler were active. Complete step here is the replayed GPU pipeline plus host submission/synchronization; trajectory output and host diagnostics are excluded.', '',f'All {len(p)} paired window/batch medians improved; minimum paired speedup {p.speedup.min():.3f}x. This is a measured result on the tested populations, not a forecast for hundreds of thousands of vehicles.', '', '## Correctness and scope','', 'CPU/comparator/radix ordering and leader tests cover inactive records, ties, signed zero, finite negative positions, non-multiple launch sizes and changing lane/road/state fixtures. Every measured replay checks exact post-motion/final Car bytes and receiving-space/reservation summaries. Nine 600-second trajectories at 0.25-second timesteps (seed 1901; identical frozen demand) match the previous milestone byte-for-byte for states, events, signals, population histories and all 180 binary snapshots. All movement/gap/conflict/state checks pass.', '', 'Additional persistent memory is 20 bytes per requested vehicle plus CUB temporary storage, allocated once before stepping. All recurring preparation and maintenance are included in timings. At 16,384 vehicles this is 538,111 additional bytes including scratch; see the artifact correctness.txt for sizes. The comparator reference can be selected with LPSIM_LEADER_SORT=comparator; radix is the default when LPSIM_LEADER_INDEX=1. The full-scan path remains available with the index disabled.', '', 'This comparison measures the complete radix implementation, including reusable scratch; it does not separately attribute savings to the sorting algorithm versus temporary-allocation reuse. It fixes construction cost, not motion-kernel tail imbalance. It provides a faster baseline for future population scaling and queue experiments. No new Nsight counter conclusions are claimed. Full-run wall times include output/diagnostics and are recorded for reproducibility, but old versus new end-to-end speedups are not inferred from separately collected full runs.', '', 'Reproduce: compile tools/leader_radix_test.cu, tools/leader_radix_replay.cu and src/simulator/scalable_junction_simulator.cu with nvcc -O3 --use_fast_math -lineinfo -arch=sm_87 -std=c++17. Place the three binaries in a fresh writable directory. Run tools/experiment_leader_radix.py --output NEW_DIRECTORY --phase snapshots, then --phase trajectories, with the recorded Jetson clock configuration. Run tools/report_leader_radix.py --artifact NEW_DIRECTORY --output NEW_REPORT_DIRECTORY. Input scenarios and snapshots are retained in the unchanged berkeley-block-imbalance-20261019 milestone.']
(r/'README.md').write_text('\n'.join(lines)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,metric,title in zip(axes,['leader_ms','wall_ms'],['Leader-index reconstruction','Complete replayed step']):
 for j,v in enumerate(['comparator','radix']):ax.bar([i+(j-.5)*.35 for i in range(3)],[summary.loc[(n,v),metric] for n in [2048,8192,16384]],width=.35,label=v)
 ax.set_xticks(range(3),['2,048','8,192','16,384']);ax.set_xlabel('Requested vehicles');ax.set_ylabel('Milliseconds');ax.set_title(title);ax.legend()
fig.tight_layout();fig.savefig(r/'timings.png',dpi=180)
with PdfPages(r/'leader-radix.pdf') as pdf:
 pdf.savefig(fig);plt.close(fig)
 fig,ax=plt.subplots(figsize=(12,5));ax.axis('off');table=summary.reset_index()[['n','variant','prepare_ms','sort_ms','link_ms','leader_ms','wall_ms']].round(4);ax.table(cellText=table.values,colLabels=table.columns,loc='center',cellLoc='center');ax.set_title('Stage timings (ms): medians across matched windows');pdf.savefig(fig);plt.close(fig)
 import textwrap
 paragraphs=[x for x in lines if x and not x.startswith('|')]
 text='\n\n'.join('\n'.join(textwrap.wrap(x.replace('#','').strip(),100)) for x in paragraphs)
 chunks=text.splitlines()
 for i in range(0,len(chunks),46):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.95,'\n'.join(chunks[i:i+46]),va='top',fontsize=10);pdf.savefig(fig);plt.close(fig)
print(summary.to_string());print('minimum paired speedup',p.speedup.min())
