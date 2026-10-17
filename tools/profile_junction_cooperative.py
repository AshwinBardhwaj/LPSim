import sys,json,subprocess,os
from pathlib import Path
sys.path.insert(0,'tools');from profile_junction_experiment import METRICS
R=Path.cwd();A=R/'artifacts/current/berkeley-cooperative-20261016';B=R/'artifacts/current/berkeley-optimized-state-20261016';N=R/'data/networks/berkeley_junction_study'
def clock(*a):subprocess.run(['sudo','-n','jetson_clocks',*a],check=True,stdout=subprocess.DEVNULL)
clock('--store',str(A/'profile-clocks.conf'))
try:
 clock()
 for label in ['distributed','concentrated']:
  name=label+'_2048_seed1401_mode0';d=A/'profiles'/name;d.mkdir(parents=True,exist_ok=True)
  cmd=['sudo','-n','env','LPSIM_MOTION_CONTEXT=3','/usr/local/cuda/bin/ncu','--cache-control','all','--clock-control','none','--kernel-name','regex:arbitrate(Cooperative|Indexed)Junctions','--launch-skip','42','--launch-count','14','--metrics',','.join(METRICS),'--export',str(d/'kernels'),str(A/'replay'),str(N/'model.txt'),str(B/'runs'/name/'0/snapshot_180.bin'),str(N/'regions.txt'),str(d/'timings.csv'),'0']
  (d/'command.json').write_text(json.dumps(cmd))
  with (d/'run.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
  with (d/'metrics.csv').open('w') as f:subprocess.run(['sudo','-n','/usr/local/cuda/bin/ncu','--import',str(d/'kernels.ncu-rep'),'--page','raw','--csv'],stdout=f,check=True)
  print(label,'profile passed',flush=True)
finally:clock('--restore',str(A/'profile-clocks.conf'))
