#!/usr/bin/env python3
"""Freeze and run workload-coverage phase; no simulator changes or allocation."""
import json,csv,random,math,hashlib,subprocess,os,shutil,argparse,time
from pathlib import Path
from collections import Counter
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-allocation-workloads-20261018';N=R/'data/networks/berkeley_junction_study';REF=R/'artifacts/current/berkeley-junction-index-20261016'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prepare():
 assert not (A/'prepared.json').exists(),'Do not overwrite frozen demand'
 A.mkdir(parents=True,exist_ok=True);g=json.loads((N/'geometry.json').read_text());roads=g['roads'];routes=json.loads((N/'route_pool.json').read_text());regions=list(map(int,(N/'regions.txt').read_text().split()))
 scores=[r['lanes']*max(5,r['speed']/.44704)/25 for r in roads]
 weights=[]
 for route in routes:
  es=route['edges'];mean=sum(scores[e]*roads[e]['length'] for e in es)/sum(roads[e]['length'] for e in es);weights.append(min(16,mean**2*math.sqrt(scores[es[0]])))
 avg=sum(weights)/len(weights);weights=[.15+.85*w/avg for w in weights]
 mass=Counter()
 for r,w in zip(routes,weights):mass[regions[r['edges'][-1]]]+=w
 hot=[k for k,v in mass.most_common(2)];hotweights=[w if regions[r['edges'][-1]] in hot else 0 for r,w in zip(routes,weights)]
 cases=[];major=[r['lanes']>=2 or r['speed']/.44704>=34.9 for r in roads]
 def major_share(selected):
  numerator=denominator=0
  for r in selected:
   for e in r['edges']:denominator+=roads[e]['length'];numerator+=roads[e]['length']*major[e]
  return numerator/denominator
 for seed in [1801,1802]:
  for pi,pattern in enumerate(['distributed','concentrated','mixed','bursty']):
   rng=random.Random(seed+pi*10000);demand=[];chosen=[]
   for i in range(16384):
    hottrip=pattern=='concentrated' and rng.random()<.8 or pattern=='mixed' and rng.random()<.5
    rid=rng.choices(range(len(routes)),weights=hotweights if hottrip else weights,k=1)[0];route=routes[rid]
    if pattern=='distributed':departure=rng.triangular(0,480,240)
    elif pattern=='concentrated':departure=rng.triangular(0,300,120)
    elif pattern=='mixed':departure=rng.triangular(60,300,180) if hottrip else rng.uniform(0,480)
    else:departure=max(0,min(480,rng.gauss(rng.choice([90,240,390]),18)))
    demand.append((departure,i,rid,hottrip));chosen.append(route)
   for n in [2048,8192,16384]:
    name=f'{pattern}_{n}_seed{seed}';d=A/'cases'/name;d.mkdir(parents=True,exist_ok=True);selected=sorted(demand[:n]);lines=[]
    with (d/'demand.csv').open('w') as f:
     writer=csv.writer(f);writer.writerow(['id','route_pool_id','departure','hot_component'])
     for vehicle,(dep,original,rid,hottrip) in enumerate(selected):
      route=routes[rid];lines.append(f'{dep:.8f} {len(route["edges"])} '+' '.join(map(str,route['edges']+route['lanes']+route['moves']+[-1])));writer.writerow([vehicle,rid,f'{dep:.8f}',int(hottrip)])
    scenario=d/'scenario.txt';scenario.write_text(str(N/'model.txt')+f'\n{n} 600 .25 0\n'+'\n'.join(lines)+'\n')
    split='train' if seed==1801 and pattern in ['distributed','concentrated'] else 'validation' if seed==1801 and pattern=='mixed' else 'test'
    cases.append(dict(name=name,n=n,pattern=pattern,seed=seed,split=split,scenario_sha256=sha(scenario),major_road_distance_share=major_share(chosen[:n]),unique_routes=len({x[2] for x in demand[:n]}),unique_origin_lanes=len({(r['edges'][0],r['lanes'][0]) for r in chosen[:n]})))
 shutil.copyfile(REF/'simulator',A/'simulator');(A/'simulator').chmod(0o755)
 files=[N/x for x in ['model.txt','geometry.json','regions.txt','route_pool.json','provenance.json']]
 dump(A/'prepared.json',dict(cases=cases,hot_destination_regions=hot,route_weight='0.15 + 0.85 * min(16,length-weighted mean(lanes*speed_mph/25)^2 * sqrt(origin lanes*speed_mph/25))/mean_weight',major_definition='lanes >=2 or speed >=35 mph; proxy, not OSM road classification',unweighted_route_pool_major_share=major_share(routes),road_major_share=sum(major)/len(major),horizon=600,dt=.25,snapshot_seconds=list(range(0,600,30)),matched_windows=[60,180,300,450],model_inputs={str(p):sha(p) for p in files},binary_sha256=sha(A/'simulator'),baseline_reference=str(REF),repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
 shutil.copyfile(R/'research/experiments/vehicle-allocation-workloads-20261018.md',A/'specification.md');print('Prepared',len(cases),'cases',flush=True)
def run():
 cfg=json.loads((A/'prepared.json').read_text());assert sha(A/'simulator')==cfg['binary_sha256'];settings=A/'clocks.conf';assert not settings.exists(),'Preserve old attempt clocks before resuming'
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 meta={'before':clock('--show'),'power_mode':subprocess.check_output(['sudo','-n','nvpmodel','-q'],text=True)};clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show')
  for c in sorted(cfg['cases'],key=lambda c:(c['n'],c['seed'],c['pattern'])):
   d=A/'runs'/c['name'];d.mkdir(parents=True,exist_ok=True)
   if (d/'validated.json').exists():continue
   scenario=A/'cases'/c['name']/'scenario.txt';assert sha(scenario)==c['scenario_sha256']
   env=dict(os.environ,LPSIM_LEADER_INDEX='1',LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_PARALLEL='1',LPSIM_JUNCTION_REQUEST_INDEX='1',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_SAVE_SNAPSHOTS='1',LPSIM_TRAJECTORY_STRIDE='120')
   cmd=[str(A/'simulator'),str(scenario)];dump(d/'command.json',dict(command=cmd,env={k:v for k,v in env.items() if k.startswith('LPSIM_')}));start=time.perf_counter()
   with (d/'run.log').open('w') as f:p=subprocess.run(cmd,cwd=d,env=env,stdout=f,stderr=subprocess.STDOUT)
   if p.returncode:dump(d/'failed.json',dict(returncode=p.returncode));raise RuntimeError(str(d))
   m=json.loads((d/'metrics.json').read_text());assert all(m[k]==0 for k in ['gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples']);assert m['completed']+m['active']+m['pending']==c['n']
   for x in csv.DictReader((d/'step_metrics.csv').open()):assert sum(int(x[k]) for k in ['pending','road_stopped','road_moving','connector','completed'])==c['n'] and int(x['not_departed'])<=int(x['pending'])
   assert len(list(d.glob('snapshot_*.bin')))==20
   dump(d/'validated.json',dict(passed=True,wall_seconds=time.perf_counter()-start,metrics=m));print(c['name'],'completed',m['completed'],'active',m['active'],'pending',m['pending'],flush=True)
  dump(A/'complete.json',dict(passed=True,cases=len(cfg['cases'])))
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');dump(A/'environment.json',meta)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);args=p.parse_args();prepare() if args.action=='prepare' else run()
