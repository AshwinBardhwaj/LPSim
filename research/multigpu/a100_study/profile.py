"""Separate Nsight replay on matched saved traffic states; no timing claims."""
from pathlib import Path
import json,subprocess,time,sys,hashlib
out=Path(sys.argv[1]);roots=sorted(Path('/artifacts').glob('study-matrix_*'));records=[]
metrics=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed','sm__warps_active.avg.pct_of_peak_sustained_active','smsp__warps_eligible.avg.per_cycle_active','smsp__thread_inst_executed_per_inst_executed.ratio','smsp__issue_active.avg.pct_of_peak_sustained_active','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_wait_per_warp_active.pct','l1tex__throughput.avg.pct_of_peak_sustained_elapsed','lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread','smsp__inst_executed.sum']
started=time.monotonic();expected={}
# All policies at the same 180/360/540 s physical states. One medium and one large demand.
for n,pattern in [(8192,'distributed'),(16384,'concentrated')]:
 case=f'train_{pattern}_{n}'
 matches=[r for r in roots if (r/(case+'__allocation')).exists()]
 if not matches:continue
 root=matches[-1];allocation=root/(case+'__allocation')
 for stamp in [180,360,540]:
  snapshot=(root/(case+'__geography__single__v1i1r-1')/'snapshot_180.bin') if stamp==180 else root/(case+'__geography__single__v0i1r7')/f'snapshot_{stamp}.bin'
  if not snapshot.exists():continue
  for policy,backend in [('geography','single'),('geography','peer'),('vehicle_count','peer'),('communication_graph','peer')]:
   name=f'{case}_{stamp}_{policy}_{backend}';d=out/name;d.mkdir()
   sys.path.insert(0,'/workspace/source/research/multigpu')
   from prepare import read_snapshot
   _,cars=read_snapshot(snapshot);geometry=json.loads(Path('/workspace/inputs/network/geometry.json').read_text());owners=list(map(int,(allocation/(policy+'.txt')).read_text().split()));pops=[]
   for rank in range(3 if backend=='peer' else 1):
    counts=dict(rank=rank,pending=0,stopped=0,moving=0,connector=0,completed=0)
    for car in cars:
     junction=geometry['roads'][car[6+car[1]]]['b']
     if backend=='peer' and owners[junction]!=rank:continue
     state=car[0];key='pending' if state==0 else ('stopped' if car[104]<.5 else 'moving') if state==1 else 'connector' if state==2 else 'completed';counts[key]+=1
    pops.append(counts)
   (d/'pre-step-populations.json').write_text(json.dumps(pops,indent=2))
   cmd=['ncu','--clock-control','none','--target-processes','all','--profile-from-start','off','--kernel-name','regex:advanceCompact|arbitrateIndexedJunctions|makeLeaderKeys|linkLaneLeaders','--launch-count','12' if backend=='peer' else '4','--metrics',','.join(metrics),'--csv','--page','raw','--log-file',str(d/'metrics.csv'),'--export',str(d/'profile'),'/workspace/bin/concurrent','/workspace/inputs/network/model.txt',str(snapshot),str(allocation/(policy+'.txt')),'1',str(d),backend,'0','0','0','0']
   t=time.monotonic()
   with (d/'run.log').open('w') as f:
    try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=50).returncode
    except subprocess.TimeoutExpired:code=-1
   if code==0:
    digest=hashlib.sha256((d/'final.bin').read_bytes()).hexdigest();key=(case,stamp)
    if key in expected and expected[key]!=digest:raise RuntimeError('Profile physical mismatch '+name)
    expected[key]=digest
   records.append(dict(case=case,time=stamp,policy=policy,backend=backend,path=name,exit_code=code,seconds=time.monotonic()-t,argv=cmd));(out/'profiles.json').write_text(json.dumps(records,indent=2));print(records[-1],flush=True)
   if time.monotonic()-started>800:sys.exit(0)
