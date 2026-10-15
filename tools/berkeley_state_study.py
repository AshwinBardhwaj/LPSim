#!/usr/bin/env python3
"""Full Berkeley research model and reproducible state-cost experiment."""
import argparse,csv,json,math,random,hashlib,subprocess,os,time
from pathlib import Path
from collections import defaultdict
import networkx as nx
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/current/berkeley-state-study-20261014'
NET=ROOT/'data/networks/berkeley_junction_study'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 assert not (NET/'model.txt').exists(), 'Do not overwrite a prepared study'
 NET.mkdir(parents=True,exist_ok=True)
 nodes={int(x['index']):x for x in csv.DictReader((ROOT/'data/networks/berkeley/nodes.csv').open())}
 edges=list(csv.DictReader((ROOT/'data/networks/berkeley/edges.csv').open()))
 def xy(n):return ((float(nodes[n]['x'])+122.29)*111320*math.cos(math.radians(37.87)),(float(nodes[n]['y'])-37.87)*111320)
 groups=nx.Graph();groups.add_nodes_from(nodes)
 for e in edges:
  if float(e['length'])<25:groups.add_edge(int(e['u']),int(e['v']))
 components=sorted(nx.connected_components(groups),key=min);owner={n:i for i,ns in enumerate(components) for n in ns}
 roads=[];outgoing=defaultdict(list)
 for e in edges:
  u,v=int(e['u']),int(e['v']);a,b=owner[u],owner[v]
  if a==b:continue
  p,q=xy(u),xy(v);length=math.dist(p,q)
  if length<=16:continue
  tangent=[(q[k]-p[k])/length for k in (0,1)]
  r=dict(id=len(roads),source_edge=int(e['uniqueid']),a=a,b=b,lanes=int(e['lanes']),length=float(e['length'])-16,speed=float(e['speed_mph'])*.44704,tangent=tangent,start=[p[k]+8*tangent[k] for k in (0,1)],end=[q[k]-8*tangent[k] for k in (0,1)])
  assert r['length']>0 and 1<=r['lanes']<=8
  roads.append(r);outgoing[a].append(r)
 def lane_point(r,l,end):
  offset=(l+.5-r['lanes']/2)*3.2;dx,dy=r['tangent'];p=r['end' if end else 'start'];return [p[0]+dy*offset,p[1]-dx*offset]
 moves=[];byjunction=defaultdict(list);graph=nx.DiGraph()
 for r in roads:
  for l in range(r['lanes']):graph.add_node((r['id'],l))
 for a in roads:
  ax,ay=a['tangent']
  for b in outgoing[a['b']]:
   bx,by=b['tangent'];dot=ax*bx+ay*by;cross=ax*by-ay*bx
   if b['b']==a['a'] or dot<-.75:continue
   turn=1 if cross>.4 else -1 if cross<-.4 else 0
   pairs=[(0,0)] if turn==1 else [(a['lanes']-1,b['lanes']-1)] if turn==-1 else [(l,l) for l in range(min(a['lanes'],b['lanes']))]
   for il,ol in pairs:
    p0,p3=lane_point(a,il,True),lane_point(b,ol,False);d=math.dist(p0,p3)
    p1=[p0[0]+ax*d*.45,p0[1]+ay*d*.45];p2=[p3[0]-bx*d*.45,p3[1]-by*d*.45]
    path=[[(1-t)**3*p0[k]+3*(1-t)**2*t*p1[k]+3*(1-t)*t*t*p2[k]+t**3*p3[k] for k in (0,1)] for t in [i/40 for i in range(41)]]
    length=LineString(path).length
    # Very short connectors are conservatively lengthened in the motion model.
    m=dict(id=len(moves),junction=a['b'],incoming=a['id'],outgoing=b['id'],in_lane=il,out_lane=ol,turn=turn,axis=int(abs(ax)>=abs(ay)),length=max(8.,length),rtor=int(turn==-1),path=path)
    moves.append(m);byjunction[a['b']].append(m['id']);graph.add_edge((a['id'],il),(b['id'],ol),weight=b['length']/b['speed']+length/7,mid=m['id'])
 shapes=[LineString(m['path']).buffer(1.2) for m in moves];conflicts=[]
 for mids in byjunction.values():
  for i in mids:
   a=moves[i]
   for j in mids:
    b=moves[j]
    if (a['incoming'],a['in_lane'])==(b['incoming'],b['in_lane']) or (a['outgoing'],a['out_lane'])==(b['outgoing'],b['out_lane']) or shapes[i].intersects(shapes[j]):conflicts.append((i,j))
 with (NET/'model.txt').open('w') as f:
  f.write(f'{len(roads)} {len(moves)} {len(components)}\n')
  for r in roads:f.write(' '.join(str(r[k]) for k in ('a','b','lanes','length','speed'))+'\n')
  for m in moves:f.write(' '.join(str(m[k]) for k in ('junction','incoming','outgoing','in_lane','out_lane','turn','axis','length','rtor'))+'\n')
  f.write(f'SPARSE {len(conflicts)}\n');f.writelines(f'{a} {b}\n' for a,b in conflicts)
 dump(NET/'geometry.json',dict(roads=roads,movements=moves,groups=[sorted(x) for x in components],conflicts=conflicts))
 # Lane-continuous routes from existing OD pairs. No hidden lane changes.
 ods=list(csv.DictReader((ROOT/'data/networks/berkeley/od_demand.csv').open()));rng=random.Random(1401);rng.shuffle(ods);routes=[];rejected=0
 target_nodes=defaultdict(list)
 for r in roads:
  for l in range(r['lanes']):target_nodes[r['b']].append((r['id'],l))
 for od in ods:
  origin,dest=owner[int(od['origin'])],owner[int(od['destination'])]
  starts=[(r['id'],l) for r in outgoing[origin] for l in range(r['lanes'])]
  targets=set(target_nodes[dest]);best=None
  if starts and targets:
   distances,paths=nx.multi_source_dijkstra(graph,starts)
   candidates=targets.intersection(distances)
   if candidates:best=paths[min(candidates,key=lambda x:(distances[x],x))]
  if not best or not 2<=len(best)<=32:rejected+=1;continue
  routes.append(dict(edges=[x[0] for x in best],lanes=[x[1] for x in best],moves=[graph[a][b]['mid'] for a,b in zip(best,best[1:])]))
  if len(routes)>=2048:break
 assert len(routes)>=512
 dump(NET/'route_pool.json',routes)
 provenance=dict(source='data/networks/berkeley',input_hashes={x:sha(ROOT/'data/networks/berkeley'/x) for x in ('nodes.csv','edges.csv','od_demand.csv')},roads=len(roads),movements=len(moves),junctions=len(components),conflict_entries=len(conflicts),route_pool=len(routes),rejected_od_pairs=rejected,assumptions=['Research network, not field calibrated','Nodes joined along <25 m edges; within-group edges removed','Endpoint chords approximate geometry because cached CSV lacks road polylines','Road lengths retain OSM values minus 16 m; connectors cubic, minimum length 8 m','Preserve up to 8 lanes; no lane changes within routes','No U-turns; inferred turn classification and lane permissions, not OSM turn restrictions','72 s synthetic signal controller at every modeled junction; right-on-red assumed permitted; no pedestrians','OD pairs retained from file; departures resampled for controlled demand; trips limited to 32 roads'])
 dump(NET/'provenance.json',provenance)
 # Hot demand concentrates departures on the most common origins in route pool.
 counts=defaultdict(int)
 for r in routes:counts[r['edges'][0]]+=1
 hot=set(sorted(counts,key=lambda x:(-counts[x],x))[:8]);hot_routes=[r for r in routes if r['edges'][0] in hot]
 cases=[]
 for seed in (1401,1402):
  for label,n,pool in [('distributed_512',512,routes),('distributed_2048',2048,routes),('concentrated_2048',2048,hot_routes)]:
   rng=random.Random(seed);demand=[]
   for i in range(n):
    r=rng.choice(pool);dep=rng.uniform(0,180);demand.append((dep,str(dep)+' '+str(len(r['edges']))+' '+' '.join(map(str,r['edges']+r['lanes']+r['moves']+[-1]))))
   demand.sort();lines=[s for _,s in demand]
   for mode in (0,2):
    name=f'{label}_seed{seed}_mode{mode}';folder=ART/'cases'/name;folder.mkdir(parents=True,exist_ok=True)
    (folder/'scenario.txt').write_text(str(NET/'model.txt')+'\n'+f'{n} 600 .25 {mode}\n'+'\n'.join(lines)+'\n')
    cases.append(dict(name=name,n=n,seed=seed,mode=mode,distribution=label,demand_sha256=hashlib.sha256(('\n'.join(lines)+'\n').encode()).hexdigest()))
 dump(ART/'prepared.json',dict(cases=cases,model_sha256=sha(NET/'model.txt'),network=provenance,horizon=600,dt=.25,profile_times=[60,180,450],repeats=3))
 print(provenance,flush=True)
def run():
 cfg=json.loads((ART/'prepared.json').read_text())
 assert json.loads((ART/'validation.json').read_text())['passed']
 # Repeat 0 supplies trajectories/pilots; only repeats 1..3 enter timing tables.
 for repeat in range(4):
  for case in cfg['cases'][::1 if repeat%2==0 else -1]:
   folder=ART/'runs'/case['name']/str(repeat);folder.mkdir(parents=True,exist_ok=True)
   if (folder/'validated.json').exists():continue
   env=dict(os.environ,LPSIM_JUNCTION_PARALLEL='1',LPSIM_LEADER_INDEX='1',LPSIM_TRAJECTORY_STRIDE='120')
   if repeat==0:env['LPSIM_SAVE_SNAPSHOTS']='1'
   start=time.perf_counter()
   with (folder/'run.log').open('w') as log:subprocess.run([str(ART/'simulator'),str(ART/'cases'/case['name']/'scenario.txt')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   metrics=json.loads((folder/'metrics.json').read_text())
   assert all(metrics[k]==0 for k in ('gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'))
   assert metrics['completed']+metrics['active']+metrics['pending']==case['n']
   for file in ('states.csv','events.csv','timeseries.csv'):
    if repeat:assert sha(folder/file)==sha(folder.parent/'0'/file),(case['name'],repeat,file)
   dump(folder/'validated.json',dict(passed=True,wall_seconds=time.perf_counter()-start,metrics=metrics))
   print(case['name'],repeat,'completed',metrics['completed'],'pending',metrics['pending'],flush=True)
def build_replays():
 # Build between timing and profiling phases, never concurrently with timing.
 source=(ROOT/'src/simulator/scalable_junction_simulator.cu').read_text().replace('#include "junction_simulator.h"','#include "'+str(ROOT/'src/simulator/junction_simulator.h')+'"').replace('#include "../../tools/leader_index.cuh"','#include "'+str(ROOT/'tools/leader_index.cuh')+'"')
 source=source.replace('namespace {','namespace {\n__device__ unsigned long long* studyCounts;',1)
 a=source.index('__global__ void advance(');b=source.index('// Deterministic junction arbitration')
 kernel=source[a:b].replace('Car c=old[i];','unsigned long long scans[2]={};auto begin=clock64();Car c=old[i];',1).replace('if(open)for(int j=0;j<n;++j){','if(open)for(int j=0;j<n;++j){scans[0]++;').replace('for(int j=0;j<n;++j){const Car& d=old[j];if(d.state==2','for(int j=0;j<n;++j){scans[1]++;const Car& d=old[j];if(d.state==2').replace('next[i]=c;','next[i]=c;studyCounts[i*3]=scans[0];studyCounts[i*3+1]=scans[1];studyCounts[i*3+2]=clock64()-begin;')
 (ART/'instrumented.cu').write_text(source[:a]+kernel+source[b:])
 harness=(ROOT/'tools/berkeley_replay.cu').read_text().replace('../src/simulator/scalable_junction_simulator.cu',str(ART/'instrumented.cu'));(ART/'instrumented_replay.cu').write_text(harness)
 for instrument in (False,True):
  name='instrumented_replay' if instrument else 'replay'
  command=['/usr/local/cuda/bin/nvcc','-O3','--use_fast_math','-lineinfo','-arch=sm_87','-std=c++17','--ptxas-options=-v']+(['-DSTUDY_INSTRUMENT',str(ART/'instrumented_replay.cu')] if instrument else [str(ROOT/'tools/berkeley_replay.cu')])+['-o',str(ART/name)]
  dump(ART/f'{name}-build-command.json',command)
  with (ART/f'{name}-build.log').open('w') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
def replay():
 # Complete any missing unprofiled repeats before starting replay/profiler work.
 run()
 build_replays()
 cfg=json.loads((ART/'prepared.json').read_text())
 for case in cfg['cases']:
  for t in (60,180,450):
   folder=ART/'replays'/case['name']/str(t);folder.mkdir(parents=True,exist_ok=True)
   snap=ART/'runs'/case['name']/'0'/f'snapshot_{t}.bin'
   env=dict(os.environ,LPSIM_STUDY_MODE=str(case['mode']))
   for selector in range(-1,22):
    out=folder/f'selector_{selector}.json'
    if out.exists():continue
    subprocess.run([str(ART/'replay'),str(NET/'model.txt'),str(snap),str(NET/'regions.txt'),str(out),str(selector),'-1'],env=env,check=True)
   for speed in (0,5):
    out=folder/f'speed_{speed}.json'
    if not out.exists():subprocess.run([str(ART/'replay'),str(NET/'model.txt'),str(snap),str(NET/'regions.txt'),str(out),'-1',str(speed)],env=env,check=True)
   controls={'shuffle':{'LPSIM_REPLAY_ORDER':'shuffle'},'lane_order':{'LPSIM_REPLAY_ORDER':'lane'},'padding':{'LPSIM_PADDING':str(case['n'])},'half_active':{'LPSIM_HALF_ACTIVE':'1'},'block64':{'LPSIM_REPLAY_BLOCK':'64'},'block256':{'LPSIM_REPLAY_BLOCK':'256'}}
   for label,settings in controls.items():
    out=folder/f'control_{label}.json'
    if not out.exists():subprocess.run([str(ART/'replay'),str(NET/'model.txt'),str(snap),str(NET/'regions.txt'),str(out),'-1','-1'],env=dict(env,**settings),check=True)
    if label!='half_active':assert json.loads(out.read_text())['physical_digest']==json.loads((folder/'selector_-1.json').read_text())['physical_digest'],label
   out=folder/'instrumented.json'
   if not out.exists():subprocess.run([str(ART/'instrumented_replay'),str(NET/'model.txt'),str(snap),str(NET/'regions.txt'),str(out),'-1','-1'],env=env,check=True)
   assert json.loads(out.read_text())['physical_digest']==json.loads((folder/'selector_-1.json').read_text())['physical_digest'], 'Instrumentation changed physical output'
   print('replay',case['name'],t,flush=True)
def controlled():
 import struct
 cfg=json.loads((ART/'prepared.json').read_text())
 for case in cfg['cases']:
  for t in (60,180,450):
   folder=ART/'replays'/case['name']/str(t);snap=ART/'runs'/case['name']/'0'/f'snapshot_{t}.bin';env=dict(os.environ,LPSIM_STUDY_MODE=str(case['mode']))
   controls={'shuffle':{'LPSIM_REPLAY_ORDER':'shuffle'},'lane_order':{'LPSIM_REPLAY_ORDER':'lane'},'padding':{'LPSIM_PADDING':str(case['n'])},'half_active':{'LPSIM_HALF_ACTIVE':'1'},'block64':{'LPSIM_REPLAY_BLOCK':'64'},'block256':{'LPSIM_REPLAY_BLOCK':'256'}}
   for label,settings in controls.items():
    out=folder/f'control_{label}.json'
    if not out.exists():subprocess.run([str(ART/'replay'),str(NET/'model.txt'),str(snap),str(NET/'regions.txt'),str(out),'-1','-1'],env=dict(env,**settings),check=True)
    if label!='half_active':assert json.loads(out.read_text())['physical_digest']==json.loads((folder/'selector_-1.json').read_text())['physical_digest'],label
   # Exact 50/50 stopped/moving mix at two active populations, unchanged total N.
   raw=snap.read_bytes();n=struct.unpack_from('i',raw)[0];assert len(raw)==8+n*428
   active=[i for i in range(n) if struct.unpack_from('i',raw,8+i*428)[0]==1];usable=len(active)//4*4
   for half in (False,True):
    label='mix_half' if half else 'mix_all';modified=bytearray(raw)
    for ordinal,i in enumerate(active):
     if ordinal>=usable or (half and ordinal%4>=2):struct.pack_into('i',modified,8+i*428,3)
     else:struct.pack_into('f',modified,8+i*428+416,5. if ordinal%2 else 0.)
    fixture=folder/f'{label}.bin';fixture.write_bytes(modified);out=folder/f'control_{label}.json'
    if not out.exists():subprocess.run([str(ART/'replay'),str(NET/'model.txt'),str(fixture),str(NET/'regions.txt'),str(out),'-1','-1'],env=env,check=True)
    result=json.loads(out.read_text());assert result['active']==2*result['stopped']
   print('controls',case['name'],t,flush=True)
def profile():
 controlled()
 from profile_junction_experiment import METRICS
 cfg=json.loads((ART/'prepared.json').read_text())
 for case in cfg['cases']:
  # Hardware samples at matched times; seed 1402 reserved for prediction testing.
  if case['seed']!=1401:continue
  for t in (60,180,450):
   folder=ART/'profiles'/case['name']/str(t);folder.mkdir(parents=True,exist_ok=True)
   if (folder/'metrics.csv').exists():continue
   lines=(ART/'cases'/case['name']/'scenario.txt').read_text().splitlines();parts=lines[1].split();parts[1]='.25';lines[1]=' '.join(parts);scenario=folder/'scenario.txt';scenario.write_text('\n'.join(lines)+'\n')
   env=dict(os.environ,LPSIM_RESTART=str(ART/'runs'/case['name']/'0'/f'snapshot_{t}.bin'),LPSIM_LEADER_INDEX='1',LPSIM_JUNCTION_PARALLEL='1')
   cmd=['sudo','-n','--preserve-env=LPSIM_RESTART,LPSIM_LEADER_INDEX,LPSIM_JUNCTION_PARALLEL','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS+['smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','launch__registers_per_thread']), '--kernel-name','regex:advance|prepareJunctions|summarizeJunctions|requestJunctionEntry|arbitrateJunctions|spawnJunctionVehicles','--cache-control','none','--clock-control','none','--launch-count','6','--force-overwrite','--export',str(folder/'kernels'),str(ART/'simulator'),str(scenario)]
   dump(folder/'command.json',dict(command=cmd,environment={k:v for k,v in env.items() if k.startswith('LPSIM_')}))
   with (folder/'ncu.log').open('w') as log:subprocess.run(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   with (folder/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'kernels.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
   print('profile',case['name'],t,flush=True)
   if t==180 and case['mode']==0 and case['n']==2048:
    source_cmd=['sudo','-n','--preserve-env=LPSIM_RESTART,LPSIM_LEADER_INDEX,LPSIM_JUNCTION_PARALLEL','/usr/local/cuda/bin/ncu','--section','SourceCounters','--import-source','yes','--kernel-name','regex:advance','--cache-control','none','--clock-control','none','--launch-count','1','--force-overwrite','--export',str(folder/'source'),str(ART/'simulator'),str(scenario)]
    with (folder/'source.log').open('w') as log:subprocess.run(source_cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (folder/'source.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'source.ncu-rep'),'--page','source','--print-source','cuda,sass','--csv'],stdout=f,check=True)
    for selector in (16,17,18,19,20):
     dest=folder/f'state_{selector}';dest.mkdir(exist_ok=True)
     state_env=dict(os.environ,LPSIM_STUDY_MODE=str(case['mode']))
     state_cmd=['sudo','-n','--preserve-env=LPSIM_STUDY_MODE','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS),'--kernel-name','regex:advance','--cache-control','none','--clock-control','none','--launch-skip','4','--launch-count','1','--force-overwrite','--export',str(dest/'kernel'),str(ART/'replay'),str(NET/'model.txt'),env['LPSIM_RESTART'],str(NET/'regions.txt'),str(dest/'output.json'),str(selector),'-1']
     with (dest/'ncu.log').open('w') as log:subprocess.run(state_cmd,env=state_env,stdout=log,stderr=subprocess.STDOUT,check=True)
     with (dest/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(dest/'kernel.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
     expected=json.loads((ART/'replays'/case['name']/str(t)/f'selector_{selector}.json').read_text())['physical_digest']
     assert json.loads((dest/'output.json').read_text())['physical_digest']==expected
 # Dynamic allocation and sparse indexing memory-safety check on a city snapshot.
 case=next(c for c in cfg['cases'] if c['name']=='distributed_2048_seed1401_mode0')
 folder=ART/'sanitizer';folder.mkdir(exist_ok=True)
 env=dict(os.environ,LPSIM_RESTART=str(ART/'runs'/case['name']/'0'/'snapshot_180.bin'),LPSIM_LEADER_INDEX='1',LPSIM_JUNCTION_PARALLEL='1')
 scenario=ART/'profiles'/case['name']/'180'/'scenario.txt'
 with (folder/'memcheck-root.log').open('w') as log:subprocess.run(['sudo','-n','--preserve-env=LPSIM_RESTART,LPSIM_LEADER_INDEX,LPSIM_JUNCTION_PARALLEL','/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','99',str(ART/'simulator'),str(scenario)],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
def quality():
 # Narrow, cold-cache retry: retain original values and label the changed protocol.
 for metrics in sorted((ART/'profiles').glob('*/*/metrics.csv')):
  rows=list(csv.DictReader(metrics.open()))[1:]
  if not any(float(r['sm__warps_active.avg.pct_of_peak_sustained_active'])>100 for r in rows):continue
  folder=metrics.parent;name=folder.parent.name;t=folder.name;dest=folder/'occupancy_retry';dest.mkdir(exist_ok=True)
  env=dict(os.environ,LPSIM_RESTART=str(ART/'runs'/name/'0'/f'snapshot_{t}.bin'),LPSIM_LEADER_INDEX='1',LPSIM_JUNCTION_PARALLEL='1')
  command=['sudo','-n','--preserve-env=LPSIM_RESTART,LPSIM_LEADER_INDEX,LPSIM_JUNCTION_PARALLEL','/usr/local/cuda/bin/ncu','--metrics','sm__warps_active.avg.pct_of_peak_sustained_active','--kernel-name','regex:prepareJunctions|spawnJunctionVehicles','--cache-control','all','--clock-control','none','--launch-count','2','--force-overwrite','--export',str(dest/'kernels'),str(ART/'simulator'),str(folder/'scenario.txt')]
  dump(dest/'command.json',command)
  with (dest/'ncu.log').open('w') as log:subprocess.run(command,cwd=dest,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  with (dest/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(dest/'kernels.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
  print('occupancy retry',name,t,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','replay','controlled','profile','quality']);a=p.parse_args();globals()[a.action]()
