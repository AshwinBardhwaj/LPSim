"""Complete padding and stopped-fraction controls under a separate fixed-clock session."""
import os,json,struct,subprocess,csv
from pathlib import Path
import optimized_state_study as study
A,B,N=study.A,study.B,study.N

def main():
 for c in json.loads((A/'prepared.json').read_text())['cases']:
  for t in (60,180,450):
   raw=(B/'runs'/c['name']/'0'/f'snapshot_{t}.bin').read_bytes();n=struct.unpack_from('i',raw)[0];active=[i for i in range(n) if struct.unpack_from('i',raw,8+428*i)[0]==1]
   for label in ('stopped90','pending_padding'):
    d=A/'extra_controls'/c['name']/str(t)/label;d.mkdir(parents=True,exist_ok=True);fixture=d/'snapshot.bin'
    data=bytearray(raw)
    if label=='stopped90':
     for k,i in enumerate(active):struct.pack_into('f',data,8+428*i+416,5. if k%10==0 else 0.)
    else:
     struct.pack_into('i',data,0,2*n);record=bytearray(raw[8:436]);struct.pack_into('i',record,0,0);struct.pack_into('f',record,408,t+1000);struct.pack_into('f',record,412,0);struct.pack_into('f',record,416,0);data.extend(record*n)
    fixture.write_bytes(data)
    for flag in (0,3):
     out=d/f'{flag}.json';env=dict(os.environ,LPSIM_MOTION_CONTEXT=str(flag),LPSIM_STUDY_MODE=str(c['mode']))
     if not out.exists():study.run([str(A/'replay'),str(N/'model.txt'),str(fixture),str(N/'regions.txt'),str(out),'-1','-1'],d/f'command_{flag}',env)
    x,y=[json.loads((d/f'{flag}.json').read_text()) for flag in (0,3)];assert x['physical_digest']==y['physical_digest']
   print('extra controls',c['name'],t,flush=True)
def quality():
 for p in sorted((A/'profiles').glob('*/*/metrics.csv')):
  rows=list(csv.DictReader(p.open()))[1:]
  invalid=any(not 0<=float(r['sm__warps_active.avg.pct_of_peak_sustained_active'].replace(',',''))<=100 for r in rows)
  if not invalid:continue
  folder=p.parent;case=folder.parent.name;t=folder.name;d=folder/'occupancy_retry';d.mkdir(exist_ok=True)
  if (d/'metrics.csv').exists():continue
  cmd=['sudo','-n','env','LPSIM_MOTION_CONTEXT=3','LPSIM_LEADER_INDEX=1','LPSIM_JUNCTION_PARALLEL=1',f'LPSIM_RESTART={A}/runs/{case}/0/snapshot_{t}.bin','/usr/local/cuda/bin/ncu','--metrics','sm__warps_active.avg.pct_of_peak_sustained_active','--kernel-name','regex:prepareJunctions|spawnJunctionVehicles','--cache-control','all','--clock-control','none','--launch-count','2','--force-overwrite','--export',str(d/'kernel'),str(A/'simulator'),str(folder/'scenario.txt')]
  study.run(cmd,d)
  with (d/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'kernel.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
  print('narrow occupancy retry',case,t,flush=True)
if __name__=='__main__':
 settings=A/'extra-original-clocks.conf';meta={}
 def clock(*args):return subprocess.check_output(['sudo','-n','jetson_clocks',*args],text=True,stderr=subprocess.STDOUT)
 assert not settings.exists();meta['before']=clock('--show');clock('--store',str(settings))
 try:clock();meta['fixed']=clock('--show');main();quality()
 finally:meta['restore']=clock('--restore',str(settings));meta['after']=clock('--show');study.dump(A/'extra-clock-environment.json',meta)
