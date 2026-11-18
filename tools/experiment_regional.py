from pathlib import Path
import json,os,subprocess
r=Path(__file__).resolve().parents[1];a=r/'artifacts/current/incremental-regional-20261023';clock=a/'regional-clocks.conf';assert not clock.exists();e={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};e.update(LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_EXCLUSIVE='0');rows=json.loads((a/'regional-prepared.json').read_text());(a/'regional').mkdir(exist_ok=True)
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True)
 for batch in range(3):
  for x in rows:
   name=f"{x['case']}_{x['t']}_{x['H']}_{batch}";cmd=[str(a/'regional_replay'),str(r/'data/networks/berkeley_junction_study/model.txt'),x['snapshot'],str(r/'data/networks/berkeley_junction_study/regions.txt'),str(a/'regional'/f'{name}.csv'),'0',str(a/'region-masks'/x['mask'])]
   with (a/'regional'/f'{name}.log').open('w') as f:q=subprocess.run(cmd,env=e,stdout=f,stderr=subprocess.STDOUT)
   if q.returncode:raise RuntimeError(name+': '+(a/'regional'/f'{name}.log').read_text())
   print(name,'exact',flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
