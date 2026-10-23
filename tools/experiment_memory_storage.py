from pathlib import Path
import subprocess,json,os,argparse
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--stage',choices=['storage','shared','shared_final'],required=True);p.add_argument('--output',type=Path,default=r/'artifacts/current/memory-policy-20261022');p.add_argument('--no-chain',action='store_true');args=p.parse_args();a=args.output.resolve();b=r/'artifacts/current/berkeley-block-imbalance-20261019';clock=a/f'{args.stage}-clocks.conf';assert not clock.exists(),'Use a fresh output or a distinct stage'
e={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};e.update(LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_MEMORY_STAGE=args.stage)
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True);(a/args.stage).mkdir(exist_ok=True)
 for batch in range(3):
  for c in json.loads((b/'prepared.json').read_text())['cases']:
   for t in [60,180,300,450]:
    cmd=[str(a/('shared_replay' if args.stage=='shared_final' else 'storage_replay')),str(r/'data/networks/berkeley_junction_study/model.txt'),str(b/'runs'/c['name']/f'snapshot_{t}.bin'),str(r/'data/networks/berkeley_junction_study/regions.txt'),str(a/args.stage/f"{c['name']}_{t}_{batch}.csv"),'0',str(b/'runs'/c['name']/f'snapshot_{t-30}.bin')]
    if batch==0 and args.stage=='storage' and not args.no_chain:cmd.append('chain')
    subprocess.run(cmd,check=True,env=e)
   print(batch,c['name'],flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
