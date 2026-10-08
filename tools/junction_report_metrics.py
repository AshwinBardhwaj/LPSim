"""Matched traffic, CUDA timing and hardware-counter appendix for the junction report."""
import csv,json,statistics,math
from pathlib import Path
import matplotlib.pyplot as plt
METRICS={
 'gpu__time_duration.sum':('Kernel duration','µs'),
 'sm__throughput.avg.pct_of_peak_sustained_elapsed':('SM throughput','% peak'),
 'sm__warps_active.avg.pct_of_peak_sustained_active':('Achieved occupancy','%'),
 'smsp__warps_eligible.avg.per_cycle_active':('Eligible warps / scheduler cycle','warps'),
 'smsp__thread_inst_executed_per_inst_executed.ratio':('Active threads / instruction','of 32'),
 'smsp__issue_active.avg.pct_of_peak_sustained_active':('Issue-active cycles','%'),
 'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct':('Long-scoreboard stall','% warp cycles'),
 'smsp__warp_issue_stalled_wait_per_warp_active.pct':('Fixed-latency wait stall','% warp cycles'),
 'smsp__warp_issue_stalled_not_selected_per_warp_active.pct':('Eligible, not selected','% warp cycles'),
 'smsp__sass_average_branch_targets_threads_uniform.pct':('Uniform branch targets','%'),
 'l1tex__throughput.avg.pct_of_peak_sustained_elapsed':('L1/TEX throughput','% peak'),
 'lts__throughput.avg.pct_of_peak_sustained_elapsed':('L2 throughput','% peak'),
}
def profiles(art):
 records=[]
 for p in sorted((art/'profiles').glob('*_*_*/kernels.csv')):
  meta=json.loads((p.parent/'provenance.json').read_text());rows=list(csv.DictReader(p.open()));units=rows.pop(0)
  for row in rows:
   data=dict(mode=meta['mode'],trips=meta['requested_trips'],time=meta['sample_time_seconds'],kernel='advance' if '::advance(' in row['Kernel Name'] else 'junctionStep',block=row['Block Size'],grid=row['Grid Size'])
   for key in METRICS:
    v=float(row[key].replace(',',''))
    if key=='gpu__time_duration.sum':v*= {'nsecond':.001,'usecond':1,'msecond':1000,'second':1e6,'ns':.001,'us':1,'µs':1,'ms':1000,'s':1e6}[units[key]]
    data[key]=v
   data['registers_per_thread']=float(row['launch__registers_per_thread'])
   data['theoretical_occupancy_pct']=float(row['sm__maximum_warps_per_active_cycle_pct'])
   records.append(data)
 return records

def add_pages(root,art,out,results,pdf,save):
 base={n:json.loads((art/'exclusive'/f'network_{n}'/'metrics.json').read_text()) for n in (20,400,1600)}
 timings=json.loads((art/'timing.json').read_text());counter=profiles(art)
 assert len(timings)==18 and len(counter)==72,(len(timings),len(counter))
 (out/'gpu-counter-samples.json').write_text(json.dumps(counter,indent=2));(out/'timing-replicates.json').write_text(json.dumps(timings,indent=2))
 with (out/'gpu-counter-samples.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=counter[0]);w.writeheader();w.writerows(counter)
 def table_page(title,cols,rows,name,caption,size=9):
  fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.15,top=.84);ax.axis('off');ax.set_title(title,fontsize=19,pad=25)
  tab=ax.table(cellText=rows,colLabels=cols,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(size);tab.scale(1,1.65 if len(rows)>15 else 2.1)
  for (r,c),cell in tab.get_celld().items():
   cell.set_edgecolor('#d1dce2')
   if r==0:cell.set_facecolor('#dcebee')
   elif r%2==0:cell.set_facecolor('#f4f7f8')
  save(fig,name,caption)
 rows=[]
 for n in (20,400,1600):
  for mode,m in [('Exclusive',base[n]),('Following',results[f'network_{n}'])]:rows.append([n,mode,m['completed'],m['active'],m['pending'],m['entries'],m['max_same_movement_occupancy']])
 table_page('Matched comparison: completion and admission', ['Trips','Controller','Completed','On roads /\nin junctions','Awaiting\nentry','Junction\nentries','Max cars on\none movement'],rows,'07-matched-results','Same 132-connector geometry, lane-continuous routes, seed 42, 72 s signals, 0.25 s steps and 600 s horizon in both modes.')
 fig,axs=plt.subplots(1,2,figsize=(11.7,8.3));fig.subplots_adjust(bottom=.18,top=.83,wspace=.32)
 for ax,n in zip(axs,(400,1600)):
  for mode,folder,color in [('Exclusive',art/'exclusive'/f'network_{n}','#ad6746'),('Following',art/f'network_{n}','#087e8b')]:
   rs=list(csv.DictReader((folder/'timeseries.csv').open()));ax.plot([float(r['time'])/60 for r in rs],[int(r['completed']) for r in rs],label=mode,color=color)
  ax.set(title=f'{n} requested trips',xlabel='Elapsed simulation minutes',ylabel='Completed routes');ax.legend();ax.grid(alpha=.2)
 fig.suptitle('Following improves discharge; high demand still overwhelms entry capacity',fontsize=17)
 save(fig,'08-completion','Demand is released during the first 180 s. Pending trips are admission backlog, not dropped trips or unfinished GPU calculations.')
 rows=[]
 for n in (20,400,1600):
  for mode in ('exclusive','platoon'):
   rs=[r for r in timings if r['trips']==n and r['mode']==mode]
   def med(k):return statistics.median(r[k] for r in rs)
   totals=[(r['advance_gpu_ms']+r['junction_gpu_ms'])/1000 for r in rs]
   rows.append([n,mode,f'{med("wall_seconds"):.3f}',f'{med("calculation_seconds"):.3f}',f'{med("advance_gpu_ms")/1000:.3f}',f'{med("junction_gpu_ms")/1000:.3f}',f'{min(totals):.3f}–{max(totals):.3f}'])
 table_page('Unprofiled timing: three complete runs per configuration',['Trips','Controller','Wall time\ns','Step + sync\ns','Road GPU\ns','Junction GPU\ns','Total GPU\nmin–max s'],rows,'09-timing','CUDA events surround each kernel; launch/event scheduling can contribute to these intervals. Wall time includes CSV output and CPU audits.')
 fig,axs=plt.subplots(1,2,figsize=(11.7,8.3));fig.subplots_adjust(bottom=.19,top=.83,wspace=.35)
 for mode,color in [('exclusive','#ad6746'),('platoon','#087e8b')]:
  xs=[20,400,1600];ys=[];lo=[];hi=[];js=[]
  for n in xs:
   rs=[r for r in timings if r['trips']==n and r['mode']==mode];vs=[(r['advance_gpu_ms']+r['junction_gpu_ms'])/1000 for r in rs];v=statistics.median(vs);ys.append(v);lo.append(v-min(vs));hi.append(max(vs)-v);js.append(statistics.median(r['junction_gpu_ms']/(r['advance_gpu_ms']+r['junction_gpu_ms'])*100 for r in rs))
  axs[0].errorbar(xs,ys,yerr=[lo,hi],marker='o',label=mode,color=color);axs[1].plot(xs,js,'o-',label=mode,color=color)
 axs[0].set(xscale='log',yscale='log',xlabel='Requested trips',ylabel='CUDA event time / 600 s run (s)',title='Workload scaling (three measured volumes)');axs[1].set(xscale='log',xlabel='Requested trips',ylabel='Junction share of GPU event time (%)',title='Serial arbitration dominates');
 for ax in axs:ax.legend();ax.grid(alpha=.2)
 ratios=[]
 for mode in ('exclusive','platoon'):
  medians=[statistics.median((r['advance_gpu_ms']+r['junction_gpu_ms'])/1000 for r in timings if r['mode']==mode and r['trips']==n) for n in (400,1600)]
  ratios.append(f"{mode}: {medians[1]/medians[0]:.2f}×")
 fig.text(.08,.095,'400 → 1,600 trips (4× demand): GPU event time grows '+', '.join(ratios)+'.',fontsize=10)
 fig.text(.08,.067,'Approximately linear over this interval; changing active/pending fractions can mask quadratic leader scans.',fontsize=10)
 save(fig,'10-scaling','Code has O(N²) road leader scans and O(N × movements) serial arbitration per step. Three mixed-state workloads do not establish an asymptotic exponent.')
 for kernel,label,index in [('advance','Parallel road propagation',11),('junctionStep','Serial junction arbitration',12)]:
  rows=[]
  keys=list(METRICS)
  for key in keys:
   row=[METRICS[key][0]+' ('+METRICS[key][1]+')']
   for n in (20,400,1600):
    vals=[]
    for mode in ('exclusive','platoon'):
     rs=[r[key] for r in counter if r['kernel']==kernel and r['trips']==n and r['mode']==mode];vals.append(f'{statistics.mean(rs):.3f}')
    row.append(' / '.join(vals))
   rows.append(row)
  table_page(label+': measured GPU and warp counters',['Metric','20: exclusive / following','400: exclusive / following','1,600: exclusive / following'],rows,f'{index:02d}-counters-{kernel}','Means of six sampled launches per kernel/configuration: two each at 60, 180 and 450 s. Raw values and units are supplied in CSV/JSON.',size=8.3)
 rows=[]
 for n in (20,400,1600):
  for mode in ('exclusive','platoon'):
   for kernel in ('advance','junctionStep'):
    rs=[r for r in counter if r['trips']==n and r['mode']==mode and r['kernel']==kernel];r=rs[0];rows.append([n,mode,kernel,r['block'],r['grid'],int(r['registers_per_thread']),f"{r['theoretical_occupancy_pct']:.1f}"])
 table_page('Kernel launch resources and theoretical occupancy',['Trips','Controller','Kernel','Threads /\nblock (xyz)','Grid\n(xyz)','Registers /\nthread','Theoretical\noccupancy %'],rows,'13-launches','Theoretical occupancy assumes enough blocks. The arbitration kernel actually launches one block containing one thread.',size=8.5)
 rows=[]
 for n in (20,400,1600):
  for mode in ('exclusive','platoon'):
   folder=art/'exclusive'/f'network_{n}' if mode=='exclusive' else art/f'network_{n}'
   series={float(r['time']):r for r in csv.DictReader((folder/'timeseries.csv').open())}
   for t in (60,180,450):
    r=series[t];rows.append([n,mode,t,r['active'],r['stopped'],r['in_junction'],r['completed'],r['pending']])
 table_page('Traffic state at the hardware profiling windows',['Trips','Controller','Time s','Active','Stopped','In box','Done','Pending'],rows,'14a-profile-states','Recorded states at each window boundary. The low-demand late sample can be empty; requested trip count is not concurrent active count.',size=8)
 fixture={}
 for mode,folder in [('exclusive',art/'exclusive/same_movement_platoon'),('platoon',art/'same_movement_platoon')]:
  times=[float(r['time']) for r in csv.DictReader((folder/'events.csv').open())];fixture[mode]=times
 fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.17,top=.85)
 for mode,color in [('exclusive','#ad6746'),('platoon','#087e8b')]:ax.step(fixture[mode],list(range(1,13)),where='post',label=mode,color=color,marker='o')
 ax.set(xlabel='Simulation seconds',ylabel='Cumulative admitted vehicles',title='Queued 12-car regression: last admission 63.50 → 43.25 seconds');ax.legend();ax.grid(alpha=.2)
 save(fig,'14b-following-fixture','Both modes complete all 12 routes. Following has up to two cars on the same connector; every step passes spacing and boundary-gap checks.')
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.92,'How to read these measurements',fontsize=23,weight='bold')
 blocks=[('Hardware and collection','Jetson Orin, compute capability 8.7, 8 SMs; 15 W nvpmodel; JetPack L4T R36.4.7.\nNsight Compute 2024.3.1; counters collected with sudo. No counter permission failure.'),('Sampling and replay','18 separate profiler runs: two controllers × three loads × three simulation times.\nFour launches per sample window (two per kernel), 11 replay passes per launch.\nProfiler wall times are excluded from benchmark timing; clocks/cache behavior can differ.'),('Warp efficiency','Active threads/instruction is measured, not inferred from branch-uniformity alone.\nThe serial junction kernel executes one active lane of a 32-lane warp by construction.\nHigh branch uniformity in that kernel does not imply efficient parallel execution.'),('Occupancy and memory','Achieved occupancy measures resident warps; SM throughput measures utilization.\nL1/TEX and L2 utilization are included. DRAM throughput is not exposed in the\navailable metric catalog for this device and is not replaced with a fabricated value.'),('Scope of the comparison','The baseline is the same physical-junction implementation with exclusive reservations.\nIt is not the original lane-map backend, a CPU baseline, or a static-partition benchmark.\nNo claim of GPU speedup or optimized warp scheduling is supported by these runs.')]
 for i,(title,body) in enumerate(blocks):y=.82-i*.151;fig.text(.065,y,title,fontsize=12,weight='bold');fig.text(.065,y-.03,body,fontsize=10.5,va='top')
 save(fig,'14-methodology','Counter definitions: https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html · Raw reports: artifacts/junction_platoons_20261007/profiles/.')
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.92,'Reproducibility and remaining model limits',fontsize=23,weight='bold')
 lines=['Inputs: data/networks/san_pablo_lane_preserving/{junction_model.json,model.txt}', 'Results: artifacts/junction_platoons_20261007/; exclusive/ contains matched baseline.', 'Every case includes scenario, demand, event/state/signal CSVs, metrics and validation.', '', 'Reproduce from the repository root:', 'python3 tools/prepare_junction_experiment.py', 'cmake --build build -j2', 'python3 tools/run_junction_experiment.py', 'python3 tools/run_junction_experiment.py --exclusive', 'python3 tools/benchmark_junction_experiment.py', 'python3 tools/profile_junction_experiment.py', 'python3 tools/report_junction_experiment.py', '', 'No lane changes in this experimental backend; demand uses lane-continuous routes.', 'When lane counts differ, through movements exist only for shared lane indices.', 'Connector speeds remain prescribed (7 m/s straight, 4 m/s turns); this is not a', 'calibrated acceleration model through the box. Conflicting swept paths remain exclusive.', 'Signals, right-on-red permissions and geometric stop lines are synthetic assumptions.', 'The high-demand backlog remains substantial. Improved completion is not field validation.']
 for i,line in enumerate(lines):fig.text(.065,.83-i*.036,line,fontsize=10.5)
 save(fig,'15-reproduce','Original October 6 datasets and PDF remain available. Source snapshots, binary hashes and raw profiling logs accompany this experiment.')
