#!/usr/bin/env python3
import json,os,subprocess,csv
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-block-imbalance-20261019';N=R/'data/networks/berkeley_junction_study'
def main():
 assert json.loads((A/'complete.json').read_text())['passed'];cfg=json.loads((A/'prepared.json').read_text());settings=A/'replay_clocks.conf';assert not settings.exists()
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True)
 meta={'before':clock('--show')};clock('--store',str(settings))
 try:
  clock();meta['fixed']=clock('--show')
  for batch in [1,2,3]:
   for c in cfg['cases'][::1 if batch%2 else -1]:
    for t in [60,180,300,450]:
     d=A/'replays'/str(batch)/c['name']/str(t);d.mkdir(parents=True,exist_ok=True)
     cmd=[str(A/'replay'),str(N/'model.txt'),str(A/'runs'/c['name']/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(d/'timings.csv'),'0'];(d/'command.json').write_text(json.dumps(dict(command=cmd,LPSIM_MOTION_CONTEXT=3)))
     with (d/'run.log').open('w') as f:subprocess.run(cmd,env=dict(os.environ,LPSIM_MOTION_CONTEXT='3'),stdout=f,stderr=subprocess.STDOUT,check=True)
     rows=list(csv.DictReader((d/'timings.csv').open()));assert len(rows)==20 and all(r['exact']=='true' for r in rows);(d/'validated.json').write_text('{"passed":true}')
    print('replay batch',batch,c['name'],'passed',flush=True)
  (A/'replay_complete.json').write_text('{"passed":true}')
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');(A/'replay_environment.json').write_text(json.dumps(meta,indent=2))
if __name__=='__main__':main()
