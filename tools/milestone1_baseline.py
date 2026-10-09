#!/usr/bin/env python3
"""Frozen paired baseline, separate timing runs and matched Nsight samples."""
import argparse,csv,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
from profile_junction_experiment import METRICS
ROOT=Path(__file__).resolve().parents[1]; ART=ROOT/'artifacts/current/milestone1-baseline-20261008'
NET=ROOT/'data/networks/milestone1_straight_corridors'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def prepare():
 assert not (ART/'manifest.json').exists(), 'Existing frozen experiment must not be overwritten'
 ART.mkdir(parents=True,exist_ok=True);NET.mkdir(parents=True,exist_ok=True)
 # 32 disconnected copies, two lanes each: capacity control, not a city model.
 roads=[];moves=[]
 for c in range(32):
  for j in range(3):roads.append([c*4+j,c*4+j+1,2,150,13.4])
  for j in range(2):
   for lane in range(2):moves.append([c*4+j+1,c*3+j,c*3+j+1,lane,lane,0,0,16,0])
 with (NET/'model.txt').open('w') as f:
  f.write(f'{len(roads)} {len(moves)} 128\n')
  for r in roads+moves:f.write(' '.join(map(str,r))+'\n')
  for i in range(len(moves)):f.write(' '.join(str(int(i==j)) for j in range(len(moves)))+'\n')
 dump(NET/'provenance.json',dict(synthetic=True,corridors=32,lanes_per_corridor=2,road_lengths_m=[150]*3,connector_length_m=16,speed_limit_mps=13.4,purpose='No crossing conflicts; replicated corridors supply exit/admission capacity at unchanged departure times. Not a Berkeley geometric comparison.'))
 frozen=ART/'inputs';frozen.mkdir(exist_ok=True)
 shutil.copy2(ROOT/'data/networks/san_pablo_lane_preserving/model.txt',frozen/'san_pablo.txt');shutil.copy2(NET/'model.txt',frozen/'corridor.txt')
 cases=[]
 for n in (20,400,1600):
  original=(ROOT/f'artifacts/junction_platoons_20261007/network_{n}/scenario.txt').read_text().splitlines()[2:]
  for mode in ('controller72','always_permitted','corridor'):
   folder=ART/'cases'/f'{mode}_{n}';folder.mkdir(parents=True,exist_ok=True)
   demand=original[:]
   if mode=='corridor':
    demand=[]
    for i,line in enumerate(original):
     c=(i%64)//2;lane=i%2;edges=[c*3+j for j in range(3)];ms=[c*4+j*2+lane for j in range(2)]
     demand.append(line.split()[0]+' 3 '+' '.join(map(str,edges+[lane]*3+ms+[-1])))
   model=frozen/('corridor.txt' if mode=='corridor' else 'san_pablo.txt')
   (folder/'scenario.txt').write_text(str(model)+'\n'+f'{n} 600 0.25 {0 if mode=="controller72" else 2}\n'+'\n'.join(demand)+'\n')
   (folder/'demand.txt').write_text('\n'.join(demand)+'\n')
   cases.append(dict(name=folder.name,n=n,mode=mode,scenario_sha256=sha(folder/'scenario.txt'),demand_sha256=sha(folder/'demand.txt'),model_sha256=sha(model)))
 dump(ART/'prepared.json',dict(cases=cases,seed=42,seed_note='Existing seeded San Pablo routes/departures reused verbatim; simulation has no random draws. Corridor lanes assigned deterministically by departure order.',dt=.25,horizon=600,profile_times=[60,180,450],repeats=3))
 print(ART)
def freeze():
 assert not (ART/'manifest.json').exists(), 'Frozen manifest already exists'
 cfg=json.loads((ART/'prepared.json').read_text());shutil.copy2(ROOT/'build/lpsim',ART/'lpsim')
 cfg.update(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),binary_sha256=sha(ART/'lpsim'),source_sha256=sha(ROOT/'src/simulator/junction_simulator.cu'),exclusive=0,threads_per_block=128,serial_junction_launch=[1,1],parallel_kernels=['advance','prepareJunctions','summarizeJunctions','requestJunctionEntry','arbitrateJunctions','spawnJunctionVehicles'])
 cfg['environment']={}
 for name,cmd in [('power',['sudo','-n','nvpmodel','-q']),('clocks',['sudo','-n','jetson_clocks','--show']),('cuda',['/usr/local/cuda/bin/nvcc','--version']),('profiler',['/usr/local/cuda/bin/ncu','--version'])]:
  r=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);cfg['environment'][name]=r.stdout
 dump(ART/'manifest.json',cfg)
def run():
 cfg=json.loads((ART/'manifest.json').read_text());assert sha(ART/'lpsim')==cfg['binary_sha256'];results=[]
 for repeat in range(3):
  for case in cfg['cases'][::1 if repeat%2==0 else -1]:
   for parallel in (1,0):
    folder=ART/'runs'/case['name']/f'{"parallel" if parallel else "serial"}_{repeat}'
    if (folder/'validated.json').exists():continue
    folder.mkdir(parents=True,exist_ok=True);env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_JUNCTION_SCENARIO=str(ART/'cases'/case['name']/'scenario.txt'),LPSIM_JUNCTION_PARALLEL=str(parallel),LPSIM_JUNCTION_EXCLUSIVE='0')
    start=time.perf_counter()
    with (folder/'run.log').open('w') as f:subprocess.run([str(ART/'lpsim')],cwd=folder,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    m=json.loads((folder/'metrics.json').read_text());m['wall_seconds']=time.perf_counter()-start
    assert all(m[k]==0 for k in ('gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'))
    assert m['trips']==m['pending']+m['active']+m['completed']
    events=list(csv.DictReader((folder/'events.csv').open()))
    assert all(int(e['light'])==2 or (int(e['event'])==2 and int(e['turn'])==-1 and int(e['light'])==0 and float(e['wait_seconds'])>=2) for e in events)
    reference=ART/'runs'/case['name']/'parallel_0'
    hashes={name:sha(folder/name) for name in ('states.csv','events.csv','signals.csv','timeseries.csv')}
    if folder!=reference:assert all(v==sha(reference/k) for k,v in hashes.items()),str(folder)
    dump(folder/'validated.json',dict(passed=True,metrics=m,hashes=hashes));print(case['name'],parallel,repeat,round(m['wall_seconds'],2),flush=True)
    # Preserve full reference trajectories and every repeat's metrics/action series.
    if folder!=reference:
     for name in hashes:(folder/name).unlink()
def profile():
 cfg=json.loads((ART/'manifest.json').read_text());assert sha(ART/'lpsim')==cfg['binary_sha256']
 for case in cfg['cases']:
  for t in cfg['profile_times']:
   folder=ART/'profiles'/case['name']/str(t);folder.mkdir(parents=True,exist_ok=True)
   if (folder/'sample.json').exists():continue
   lines=(ART/'cases'/case['name']/'scenario.txt').read_text().splitlines();fields=lines[1].split();fields[1]=str(t+.25);lines[1]=' '.join(fields);(folder/'scenario.txt').write_text('\n'.join(lines)+'\n')
   cmd=['sudo','-n','env',f'LPSIM_JUNCTION_SCENARIO={folder}/scenario.txt','LPSIM_JUNCTION_PARALLEL=1','LPSIM_JUNCTION_EXCLUSIVE=0','/usr/local/cuda/bin/ncu','--metrics',','.join(METRICS),'--launch-skip',str(t*4*6),'--launch-count','6','--force-overwrite','--export',str(folder/'kernels'),str(ART/'lpsim')]
   dump(folder/'command.json',cmd)
   with (folder/'ncu.log').open('w') as f:subprocess.run(cmd,cwd=folder,stdout=f,stderr=subprocess.STDOUT,check=True)
   with (folder/'kernels.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(folder/'kernels.ncu-rep'),'--csv','--page','raw'],stdout=f,check=True)
   rows=list(csv.DictReader((folder/'step_metrics.csv').open()));state=next(r for r in rows if float(r['time'])==t)
   ref=next(r for r in csv.DictReader((ART/'runs'/case['name']/'parallel_0/step_metrics.csv').open()) if float(r['time'])==t)
   assert all(state[k]==v for k,v in ref.items() if not k.endswith('_gpu_ms'))
   dump(folder/'sample.json',dict(time=t,window_end=t+.25,pre_step_population_and_step_actions=state,matched_reference=True));print('profile',case['name'],t,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','freeze','run','profile']);a=p.parse_args();globals()[a.action]()
