#!/usr/bin/env python3
import argparse,csv,hashlib,json,os,random,subprocess,statistics
from pathlib import Path
from profile_junction_experiment import METRICS
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/current/milestone2-snapshots-20261009'
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 assert not (ART/'manifest.json').exists()
 edges=list(csv.DictReader((ROOT/'data/networks/berkeley/edges.csv').open()));outgoing={}
 for i,e in enumerate(edges):outgoing.setdefault(e['u'],[]).append(i)
 eligible=[];roads=[];moves=[]
 for i,e in enumerate(edges):roads.append([int(e['u']),int(e['v']),int(e['lanes']),float(e['length']),float(e['speed_mph'])*.44704])
 for i,e in enumerate(edges):
  options=[j for j in outgoing.get(e['v'],[]) if edges[j]['v']!=e['u']]
  if not options:continue
  j=min(options);mid=len(moves);moves.append([int(e['v']),i,j,0,0,0,0,16,0])
  if float(e['length'])>=100:
   for pos in range(30,int(float(e['length'])-30),20):eligible.append((i,float(pos),mid))
 # Spread nested active subsets across the city, preserving every position when padded/permuted.
 random.Random(20261009).shuffle(eligible);assert len(eligible)>=2048
 with (ART/'model.txt').open('w') as f:
  f.write(f'{len(roads)} {len(moves)}\n')
  for r in roads+moves:f.write(' '.join(map(str,r))+'\n')
 canonical=ROOT/'data/networks/berkeley_snapshot_reference';canonical.mkdir(exist_ok=True)
 __import__('shutil').copy2(ART/'model.txt',canonical/'model.txt')
 dump(canonical/'provenance.json',dict(source='data/networks/berkeley',scope='Road table copied numerically; one connected outgoing edge per eligible road; lane 0 synthetic snapshot fixtures, not calibrated intersection geometry.'))
 cases=[]
 def make(name,active,total,stopped,padding='pending',order='id',group='movement'):
  rows=[]
  for i,(edge,pos,mid) in enumerate(eligible[:active]):
   # Interleave stopped labels through stable IDs; deterministic fraction exact for .5.
   isstop=(i*37%active)<round(active*stopped)
   rows.append([i,1,edge,pos,0 if isstop else 5,mid])
  edge,pos,mid=eligible[0]
  rows += [[i,0 if padding=='pending' else 3,edge,0,0,mid] for i in range(active,total)]
  if order=='shuffle':random.Random(17).shuffle(rows)
  elif order=='lane':rows.sort(key=lambda r:(r[1]!=1,r[2],r[3],r[0]))
  elif order=='interleave':rows.sort(key=lambda r:(r[0]%active,r[0]//active))
  elif order=='state':rows.sort(key=lambda r:(r[1],r[4],r[2],r[3]))
  file=ART/'snapshots'/f'{name}.txt';file.parent.mkdir(exist_ok=True);file.write_text(str(total)+'\n'+'\n'.join(' '.join(map(str,r)) for r in rows)+'\n')
  cases.append(dict(name=name,active=active,total=total,stopped_fraction=stopped,padding=padding,order=order,group=group,sha256=sha(file)))
 for stop in (0,.5,.9,1):make(f'motion_{int(stop*100)}',2048,2048,stop)
 for active in (256,512,1024):make(f'population_{active}',active,active,.5,group='population')
 for padding in ('pending','completed'):
  for total in (4096,8192):make(f'padding_{padding}_{total}',2048,total,.5,padding=padding,group='padding')
 for order in ('shuffle','lane','interleave','state'):make(f'order_{order}',2048,4096,.5,order=order,group='order')
 dump(ART/'manifest.json',dict(cases=cases,roads=len(roads),movements=len(moves),available_slots=len(eligible),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sha(ROOT/'src/simulator/junction_simulator.cu'),edges_sha256=sha(ROOT/'data/networks/berkeley/edges.csv'),nodes_sha256=sha(ROOT/'data/networks/berkeley/nodes.csv'),model_sha256=sha(ART/'model.txt'),dt=.25,time=180,controller_mode=2,block=128,seed=20261009,scope='Unchanged production advance kernel only. Full Berkeley road table; lane 0 fixtures, two-edge routes, no connector occupants or stop-line crossings. Junction pipeline fixed buffers cannot represent full Berkeley.',repeats=3,steps_per_repeat=20,warmup_steps=3))
def build():
 src=(ROOT/'src/simulator/junction_simulator.cu').read_text();part=src[src.index('__global__ void advance'):src.index('// Deterministic junction arbitration')]
 part=part.replace('Car c=old[i];','unsigned long long scans[3]={};Car c=old[i];')
 part=part.replace('if(open)for(int j=0;j<n;++j){','if(open)for(int j=0;j<n;++j){scans[0]++;')
 part=part.replace('if(j==i)continue;const Car& d=', 'if(j==i)continue;scans[1]++;const Car& d=')
 part=part.replace('for(int j=0;j<n;++j){const Car& d=old[j];if(d.state==2', 'for(int j=0;j<n;++j){scans[2]++;const Car& d=old[j];if(d.state==2')
 part=part.replace('next[i]=c;', 'next[i]=c;if(old[i].state==1){atomicAdd(snapshotCounts,1ULL);for(int k=0;k<3;++k)atomicAdd(snapshotCounts+k+1,scans[k]);if(roads[old[i].edges[old[i].k]].length-old[i].pos<=.11f)atomicAdd(snapshotCounts+4,1ULL);}')
 inst=src[:src.index('__global__ void advance')]+part+src[src.index('// Deterministic junction arbitration'):];inst=inst.replace('namespace {','namespace {\n__device__ unsigned long long snapshotCounts[5];',1)
 instfile=ART/'instrumented_kernel.cu';instfile.write_text(inst)
 harness=(ROOT/'tools/snapshot_benchmark.cu').read_text().replace('../src/simulator/junction_simulator.cu',str(instfile));(ART/'instrumented_harness.cu').write_text(harness)
 for instrument in (False,True):
  cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_87','-std=c++17','-I'+str(ROOT/'src/simulator'),'--ptxas-options=-v']
  if instrument:cmd+=['-DSNAPSHOT_INSTRUMENT',str(ART/'instrumented_harness.cu')]
  else:cmd+=[str(ROOT/'tools/snapshot_benchmark.cu')]
  cmd+=['-o',str(ART/('instrumented' if instrument else 'benchmark'))]
  with (ART/('instrumented-build.log' if instrument else 'build.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
 dump(ART/'binaries.json',{name:sha(ART/name) for name in ('benchmark','instrumented')})
def run():
 cfg=json.loads((ART/'manifest.json').read_text())
 for repeat in range(3):
  for case in cfg['cases'][::1 if repeat%2==0 else -1]:
   folder=ART/'runs'/case['name'];folder.mkdir(parents=True,exist_ok=True);output=folder/f'{repeat}.csv'
   subprocess.run([str(ART/'benchmark'),str(ART/'model.txt'),str(ART/'snapshots'/f"{case['name']}.txt"),str(output),'20'],check=True)
   if repeat:assert sha(output)==sha(folder/'0.csv')
   else:subprocess.run([str(ART/'instrumented'),str(ART/'model.txt'),str(ART/'snapshots'/f"{case['name']}.txt"),str(folder/'instrumented.csv'),'1'],check=True);assert sha(folder/'instrumented.csv')==sha(output)
   print(case['name'],repeat,flush=True)
 dump(ART/'runs-complete.json',dict(passed=True))
def profile():
 extra=['smsp__inst_executed.sum','l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum','lts__t_sectors_op_read.sum','launch__registers_per_thread','launch__local_mem_per_thread','l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum','l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum']
 cfg=json.loads((ART/'manifest.json').read_text())
 for case in cfg['cases']:
  folder=ART/'profiles'/case['name'];folder.mkdir(parents=True,exist_ok=True)
  cmd=['sudo','-n','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS+extra),'--launch-skip','3','--launch-count','1','--force-overwrite','--export',str(folder/'kernel'),str(ART/'benchmark'),str(ART/'model.txt'),str(ART/'snapshots'/f"{case['name']}.txt"),str(folder/'output.csv'),'1']
  dump(folder/'command.json',cmd)
  with (folder/'ncu.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
  with (folder/'metrics.csv').open('w') as out:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'kernel.ncu-rep'),'--page','raw','--csv'],stdout=out,check=True)
  assert sha(folder/'output.csv')==sha(ART/'runs'/case['name']/'0.csv');print('profile',case['name'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','build','run','profile']);a=p.parse_args();globals()[a.action]()
