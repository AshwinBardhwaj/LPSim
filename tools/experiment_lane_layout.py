"""Run paired physical lane-layout tests; restore Jetson clocks on failure too."""
from pathlib import Path
import argparse,subprocess,json,os,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'artifacts/current/lane-layout-20261022');args=p.parse_args();a=args.output.resolve();a.mkdir(parents=True,exist_ok=True)
b=ROOT/'artifacts/current/berkeley-block-imbalance-20261019';cases=json.loads((b/'prepared.json').read_text())['cases']
env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_EXCLUSIVE='0')
clock=a/'original-clocks.conf';assert not clock.exists(),'Use a fresh output directory'
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True)
 (a/'clocks.txt').write_text(subprocess.check_output(['sudo','-n','jetson_clocks','--show'],text=True))
 (a/'replays').mkdir(exist_ok=True)
 for batch in range(3):
  for c in cases:
   for t in [60,180,300,450]:
    out=a/'replays'/f"{c['name']}_{t}_{batch}.csv"
    cmd=[str(a/'replay'),str(ROOT/'data/networks/berkeley_junction_study/model.txt'),str(b/'runs'/c['name']/f'snapshot_{t}.bin'),str(ROOT/'data/networks/berkeley_junction_study/regions.txt'),str(out),'0']
    subprocess.run(cmd,env=env,check=True)
   print(batch,c['name'],flush=True)
finally:
 subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
