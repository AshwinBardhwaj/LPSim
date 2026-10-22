from pathlib import Path
import subprocess,json,os,argparse
r=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=r/'artifacts/current/memory-policy-20261022');args=parser.parse_args();a=args.output.resolve();b=r/'artifacts/current/berkeley-block-imbalance-20261019';clock=a/'compact-clocks.conf'
e={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};e.update(LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_EXCLUSIVE='0')
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True);(a/'clocks.txt').write_text(subprocess.check_output(['sudo','-n','jetson_clocks','--show'],text=True));(a/'compact').mkdir(exist_ok=True)
 for batch in range(3):
  for c in json.loads((b/'prepared.json').read_text())['cases']:
   for t in [60,180,300,450]:
    subprocess.run([str(a/'compact_replay'),str(r/'data/networks/berkeley_junction_study/model.txt'),str(b/'runs'/c['name']/f'snapshot_{t}.bin'),str(r/'data/networks/berkeley_junction_study/regions.txt'),str(a/'compact'/f"{c['name']}_{t}_{batch}.csv"),'0'],check=True,env=e)
   print(batch,c['name'],flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
