#!/usr/bin/env python3
import json,os,subprocess,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-oracle-20261016';B=R/'artifacts/current/berkeley-optimized-state-20261016';N=R/'data/networks/berkeley_junction_study'
def main():
 env=dict(os.environ,LPSIM_MOTION_CONTEXT='3');clock=A/'clocks.conf';assert not clock.exists()
 def clocks(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True)
 metadata={'before':clocks('--show')};clocks('--store',str(clock))
 try:
  clocks();metadata['fixed']=clocks('--show')
  for c in json.loads((B/'prepared.json').read_text())['cases']:
   for t in [60,180,450]:
    d=A/c['name']/str(t);d.mkdir(parents=True,exist_ok=True);cmd=[str(A/'replay'),str(N/'model.txt'),str(B/'runs'/c['name']/'0'/f'snapshot_{t}.bin'),str(N/'regions.txt'),str(d/'timings.csv'),str(c['mode'])]
    (d/'command.json').write_text(json.dumps(cmd));
    with (d/'run.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    print(c['name'],t,'passed',flush=True)
  (A/'complete.json').write_text('{"passed":true}')
 finally:clocks('--restore',str(clock));metadata['after']=clocks('--show');(A/'environment.json').write_text(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
