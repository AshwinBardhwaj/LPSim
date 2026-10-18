#!/usr/bin/env python3
import json,os,subprocess,csv,hashlib,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/vehicle-allocation-oracle-20261018';B=R/'artifacts/current/berkeley-allocation-workloads-20261018';N=R/'data/networks/berkeley_junction_study'
def main():
 cfg=json.loads((B/'prepared.json').read_text());shutil.copyfile(R/'research/experiments/vehicle-allocation-oracle-20261018.md',A/'protocol.md');(A/'cases.json').write_text(json.dumps(cfg['cases'],indent=2))
 settings=A/'clocks.conf';assert not settings.exists()
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 env=dict(os.environ,LPSIM_MOTION_CONTEXT='3');meta={'before':clock('--show'),'power_mode':subprocess.check_output(['sudo','-n','nvpmodel','-q'],text=True)};clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show')
  for batch in [1,2,3]:
   for c in cfg['cases'][::1 if batch%2 else -1]:
    for t in [60,180,300,450]:
     d=A/'runs'/str(batch)/c['name']/str(t);d.mkdir(parents=True,exist_ok=True)
     if (d/'validated.json').exists():continue
     cmd=[str(A/'replay'),str(N/'model.txt'),str(B/'runs'/c['name']/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(d/'timings.csv'),'0'];(d/'command.json').write_text(json.dumps(dict(command=cmd,LPSIM_MOTION_CONTEXT=3)))
     with (d/'run.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
     rows=list(csv.DictReader((d/'timings.csv').open()));assert len(rows)==140 and all(x['exact']=='true' for x in rows);(d/'validated.json').write_text('{"passed":true}')
    print('batch',batch,c['name'],'passed',flush=True)
  (A/'complete.json').write_text('{"passed":true}')
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');(A/'environment.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
