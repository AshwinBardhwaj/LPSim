#!/usr/bin/env python3
"""Population and spatial workload coverage; no regional timing attribution."""
import csv,json,math,textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-allocation-workloads-20261018';O=R/'reports/berkeley-allocation-workloads-20261018';N=R/'data/networks/berkeley_junction_study'
DT=np.dtype(dict(names=['state','k','mid','denial','edges','lanes','departure','pos','speed'],formats=['i4','i4','i4','i4',('i4',32),('i4',32),'f4','f4','f4'],offsets=[0,4,12,20,24,152,408,412,416],itemsize=428))
def read(p):return json.loads(p.read_text())
def write(name,rows):
 if not rows:return
 with (O/(name+'.csv')).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 O.mkdir(exist_ok=True);cfg=read(A/'prepared.json');g=read(N/'geometry.json');roads=g['roads'];moves=g['movements'];regions=np.array(list(map(int,(N/'regions.txt').read_text().split())));length=np.array([r['length'] for r in roads]);major=np.array([r['lanes']>=2 or r['speed']/.44704>=34.9 for r in roads]);summaries=[];windows=[];spatial=[];series={}
 for c in cfg['cases']:
  d=A/'runs'/c['name'];assert read(d/'validated.json')['passed'];m=read(d/'metrics.json');steps=list(csv.DictReader((d/'step_metrics.csv').open()));ss={k:np.array([float(r[k]) for r in steps]) for k in steps[0]};series[c['name']]=ss
  row=dict(case=c['name'],pattern=c['pattern'],seed=c['seed'],split=c['split'],requested=c['n'],peak_road_active=int(max(ss['road_moving']+ss['road_stopped'])),peak_moving=int(max(ss['road_moving'])),mean_moving=float(np.mean(ss['road_moving'])),mean_stopped=float(np.mean(ss['road_stopped'])),mean_connector=float(np.mean(ss['connector'])),mean_eligible_pending=float(np.mean(ss['pending']-ss['not_departed'])),mean_future=float(np.mean(ss['not_departed'])),completed=m['completed'],final_active=m['active'],final_pending=m['pending'],entered=c['n']-m['pending'],entry_rate_vps=(c['n']-m['pending'])/600,completion_rate_vps=m['completed']/600,gpu_total_ms=m['advance_gpu_ms']+m['junction_gpu_ms'],advance_with_shared_indexes_ms=m['advance_gpu_ms'],junction_pipeline_ms=m['junction_gpu_ms'],calculation_seconds=m['calculation_seconds'],run_wall_seconds=read(d/'validated.json')['wall_seconds'],planned_major_road_distance_share=c['major_road_distance_share'],unique_origin_lanes=c['unique_origin_lanes']);summaries.append(row)
  for p in sorted(d.glob('snapshot_*.bin'),key=lambda p:int(p.stem.split('_')[1])):
   raw=p.read_bytes();count=int(np.frombuffer(raw,dtype='i4',count=1)[0]);t=float(np.frombuffer(raw,dtype='f4',count=1,offset=4)[0]);v=np.frombuffer(raw,dtype=DT,offset=8);assert len(v)==count==c['n']
   road=v['state']==1;con=v['state']==2;pending=v['state']==0;future=pending&(v['departure']>t);done=v['state']==3;stopped=road&(v['speed']<.5);moving=road&~stopped
   edge=v['edges'][np.arange(count),v['k']];reg=regions[edge];lane=v['lanes'][np.arange(count),v['k']];roadids=np.flatnonzero(road);leaders=np.zeros(count,dtype=bool)
   if len(roadids):
    key=edge[roadids]*8+lane[roadids];order=np.lexsort((v['pos'][roadids],key));sortedids=roadids[order];sortedkey=key[order];last=np.r_[sortedkey[1:]!=sortedkey[:-1],True];leaders[sortedids[last]]=True
   near=road&(length[edge]-v['pos']<=30);queued_follower=stopped&~leaders
   active_counts=[];fractions=[];qualified=[]
   for region in range(16):
    mask=reg==region;nr=int(np.count_nonzero(road&mask));ns=int(np.count_nonzero(stopped&mask));nm=int(np.count_nonzero(moving&mask));nc=int(np.count_nonzero(con&mask));active_counts.append(nr+nc)
    if nr:fractions.append(ns/nr)
    if nr>=50:qualified.append(ns/nr)
    spatial.append(dict(case=c['name'],time=t,region=region,road_active=nr,road_stopped=ns,road_moving=nm,connector=nc,eligible_pending=int(np.count_nonzero(pending&~future&mask)),future=int(np.count_nonzero(future&mask)),leaders=int(np.count_nonzero(leaders&mask)),near_end_leaders=int(np.count_nonzero(leaders&near&mask)),queued_followers=int(np.count_nonzero(queued_follower&mask)),previous_step_space_denials=int(np.count_nonzero((v['denial']==2)&mask&road)),mean_road_speed=float(np.mean(v['speed'][road&mask])) if nr else 0,stopped_fraction=ns/nr if nr else 0))
   speeds=v['speed'][road];qs=np.quantile(speeds,[.1,.5,.9]) if len(speeds) else [0,0,0];cv=float(np.std(active_counts)/np.mean(active_counts)) if sum(active_counts) else 0
   windows.append(dict(case=c['name'],pattern=c['pattern'],seed=c['seed'],split=c['split'],requested=count,time=t,road_stopped=int(np.count_nonzero(stopped)),road_moving=int(np.count_nonzero(moving)),connector=int(np.count_nonzero(con)),road_active=int(np.count_nonzero(road)),active=int(np.count_nonzero(road|con)),eligible_pending=int(np.count_nonzero(pending&~future)),future=int(np.count_nonzero(future)),completed=int(np.count_nonzero(done)),leaders=int(np.count_nonzero(leaders)),near_end_leaders=int(np.count_nonzero(leaders&near)),queued_followers=int(np.count_nonzero(queued_follower)),speed_p10=float(qs[0]),speed_median=float(qs[1]),speed_p90=float(qs[2]),mean_speed=float(np.mean(speeds)) if len(speeds) else 0,nonempty_regions=sum(x>0 for x in active_counts),regional_active_cv=cv,largest_region_share=max(active_counts)/sum(active_counts) if sum(active_counts) else 0,qualified_regions=len(qualified),qualified_stopped_range=max(qualified)-min(qualified) if qualified else 0,major_road_vehicle_share=float(np.mean(major[edge[road]])) if np.count_nonzero(road) else 0))
 matched=[r for r in windows if r['time'] in cfg['matched_windows']];growth=[]
 for x in matched:
  if x['requested']==2048:continue
  base=next(y for y in matched if y['requested']==2048 and all(y[k]==x[k] for k in ['pattern','seed','time']));growth.append(dict(pattern=x['pattern'],seed=x['seed'],time=x['time'],requested=x['requested'],active=x['active'],reference_active=base['active'],ratio=x['active']/base['active'] if base['active'] else None))
 families={x['pattern'] for x in growth if x['ratio'] is not None and x['ratio']>=3};hetero=[r for r in matched if r['qualified_regions']>=4 and r['qualified_stopped_range']>=.25];movingmax=max(r['road_moving'] for r in matched)
 gate=dict(population_growth_pass=len(families)>=2,growth_families=sorted(families),regional_heterogeneity_pass=bool(hetero),qualifying_windows=len(hetero),moving_population_pass=movingmax>=500,max_matched_moving=movingmax,correctness_pass=True,passed=len(families)>=2 and bool(hetero) and movingmax>=500,caveat='Population/state coverage gate only; no measured regional compute-cost or GPU saturation claim')
 write('runs',summaries);write('snapshots',windows);write('regions',spatial);write('matched_growth',growth);(O/'gate.json').write_text(json.dumps(gate,indent=2));(O/'specification.md').write_text((A/'specification.md').read_text())
 notes=[f"Workload coverage gate: {'PASS' if gate['passed'] else 'FAIL'}. Growth >=3x occurs in {len(families)} demand families. {len(hetero)} matched windows have at least four populated regions and >=0.25 spread in stopped fractions. Maximum matched moving population: {movingmax}.",
 'Road importance uses lanes and speed, not unavailable highway classifications or measured Berkeley counts. Demand is synthetic with plausible topology weighting and randomized departures; it is not field calibrated. Existing inferred lane-continuous routes and synthetic signals remain limitations.',
 'Future departures and eligible-but-pending trips are separate. Active counts exclude both. Larger requested counts are not interpreted as more active traffic without the observed population evidence.',
 'Regional populations, leaders and queues are workload features, not regional execution time. No allocator was built in this phase. The next gate must measure task costs and compare common execution paths. The saved advance timing includes leader sorting and motion-context construction; it cannot be called pure vehicle-motion time.',
 'Timing is exploratory: one full run per case with fixed clocks. CUDA stage totals exclude host validation/output. Host elapsed time includes correctness checks and output and is not simulation-kernel performance. No speedup claims. All simulations passed built-in gap, conflict and state checks and population conservation.',
 'Patterns intentionally vary both route concentration and departure timing. They test workload coverage, not the isolated causal effect of congestion or density. Long routes, synthetic signals and lane-continuity constraints also affect completion.',
 'Thresholds and training/validation/test partitions were recorded before execution. Descriptive test-set coverage is reported here; future learned-policy fitting must not use test outcomes. Steps 1–2 do not establish cross-network generalization or optimal allocation.'
 ]
 empty_regions=[region for region in range(16) if not any(r['road_active'] or r['connector'] for r in spatial if r['region']==region)]
 notes.append(f'Geographic coverage is uneven. Regions with zero observed active traffic across all saved windows: {empty_regions}. The reserved region group 12–15 must not be treated as strong geographic generalization evidence merely because it was held out; report its actual population coverage before fitting.')
 notes.append('Completion remains low at the largest demand: 436–664 of 16,384 trips finish in 600 s. These cases provide moving-plus-queued workload coverage, not a demonstrated free-flow control. Congestion effects cannot be separated from route length, arrival timing and the frozen signal model using these aggregate results alone.')
 notes.append(f"Major-road proxies are {100*cfg['road_major_share']:.1f}% of modeled roads. They represent {100*cfg['unweighted_route_pool_major_share']:.1f}% of unweighted route-pool distance versus {100*np.mean([c['major_road_distance_share'] for c in cfg['cases']]):.1f}% of sampled planned distance. This verifies the weighting effect; it does not validate the demand against observed traffic.")
 (O/'conclusions.md').write_text('\n\n'.join(notes));
 with PdfPages(O/'allocation-workloads.pdf') as pdf:
  fig=plt.figure(figsize=(12,10));fig.text(.05,.96,'Vehicle allocation: specification and workload coverage',fontsize=17,va='top');fig.text(.05,.9,'\n\n'.join(textwrap.fill(x,110) for x in notes),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
  for seed in [1801,1802]:
   rr=[r for r in summaries if r['seed']==seed];fig,ax=plt.subplots(figsize=(15,8));ax.axis('off');ax.set_title(f'Seed {seed}: actual populations and outcomes over 600 seconds');t=ax.table(cellText=[[r['pattern'],r['requested'],r['peak_road_active'],r['peak_moving'],round(r['mean_stopped']),r['final_pending'],r['completed'],f"{r['gpu_total_ms']/1000:.2f}"] for r in rr],colLabels=['Pattern','Requested','Peak road active','Peak moving','Mean stopped','Pending at 600s','Completed','GPU stage sum (s)'],loc='center');t.auto_set_font_size(False);t.set_fontsize(9);t.scale(1,1.8);fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'table_{seed}.png',dpi=140);plt.close(fig)
  for seed in [1801,1802]:
   fig,axes=plt.subplots(2,2,figsize=(13,8))
   for ax,pattern in zip(axes.flat,['distributed','concentrated','mixed','bursty']):
    for n in [2048,8192,16384]:
     ss=series[f'{pattern}_{n}_seed{seed}'];ax.plot(ss['time'],ss['road_moving']+ss['road_stopped']+ss['connector'],label=str(n))
    ax.set_title(pattern);ax.set_xlabel('Simulation seconds');ax.set_ylabel('On-road + connector vehicles');ax.legend()
   fig.suptitle(f'Actual active population, seed {seed}');fig.tight_layout();pdf.savefig(fig);fig.savefig(O/f'active_{seed}.png',dpi=140);plt.close(fig)
  fig,axes=plt.subplots(2,2,figsize=(13,8));bins=np.arange(0,601,30)
  for ax,pattern in zip(axes.flat,['distributed','concentrated','mixed','bursty']):
   name=f'{pattern}_16384_seed1801';demand=list(csv.DictReader((A/'cases'/name/'demand.csv').open()));departures=np.array([float(x['departure']) for x in demand]);planned,_=np.histogram(departures,bins);ss=series[name];entered,_=np.histogram(ss['time'],bins,weights=ss['spawned']);ax.step(bins[:-1],planned,where='post',label='Planned departures');ax.step(bins[:-1],entered,where='post',label='Actual entries');ax.set_title(pattern);ax.set_xlabel('Simulation seconds');ax.set_ylabel('Vehicles per 30 seconds');ax.legend()
  fig.suptitle('Requested demand versus actual admission to roads');fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'departures_entries.png',dpi=140);plt.close(fig)
  patterns=['distributed','concentrated','mixed','bursty'];z=np.full((4,16),np.nan);counts=np.zeros((4,16),dtype=int)
  for i,pattern in enumerate(patterns):
   rr=[r for r in spatial if r['case']==f'{pattern}_16384_seed1802' and r['time']==300]
   for r in rr:
    counts[i,r['region']]=r['road_active']
    if r['road_active']:z[i,r['region']]=r['stopped_fraction']
  fig,ax=plt.subplots(figsize=(15,5));im=ax.imshow(z,vmin=0,vmax=1,cmap='YlOrRd',aspect='auto');ax.set_xticks(range(16));ax.set_yticks(range(4),patterns);ax.set_xlabel('Region ID');ax.set_title('16,384 trips, held-out seed, t=300 s: color = stopped fraction; text = road-active vehicles')
  for i in range(4):
   for j in range(16):ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=8,color='white' if z[i,j]>.65 else 'black')
  fig.colorbar(im,ax=ax,label='Stopped fraction');fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'regional_states.png',dpi=140);plt.close(fig)
  fig,ax=plt.subplots(figsize=(10,6))
  for pattern in ['distributed','concentrated','mixed','bursty']:
   rr=[r for r in matched if r['pattern']==pattern];ax.scatter([r['active'] for r in rr],[r['regional_active_cv'] for r in rr],label=pattern,alpha=.6)
  ax.set_xlabel('Active vehicles in matched snapshot');ax.set_ylabel('Regional active-count coefficient of variation');ax.set_title('Spatial imbalance is a workload proxy, not measured GPU imbalance');ax.legend();fig.tight_layout();pdf.savefig(fig);fig.savefig(O/'regional_variation.png',dpi=140);plt.close(fig)
 (O/'index.html').write_text('<h1>Vehicle-allocation workload study</h1><p>'+notes[0]+'</p><a href="allocation-workloads.pdf">PDF</a> | <a href="specification.md">Frozen specification</a> | <a href="runs.csv">Run metrics</a> | <a href="snapshots.csv">Snapshot metrics</a> | <a href="regions.csv">Regional metrics</a>'+''.join('<img width="100%" src="'+p.name+'">' for p in sorted(O.glob('*.png'))))
 print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
