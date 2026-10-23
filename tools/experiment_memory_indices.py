from pathlib import Path
import subprocess,json,os,argparse
r=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--phase',choices=['characterize','evaluate'],required=True);p.add_argument('--output',type=Path,default=r/'artifacts/current/memory-policy-20261022');args=p.parse_args();a=args.output.resolve();b=r/'artifacts/current/berkeley-block-imbalance-20261019';name='indices' if args.phase=='characterize' else 'allocation';clock=a/f'{name}-clocks.conf';assert not clock.exists(),'Use fresh output'
e={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};e.update(LPSIM_MOTION_CONTEXT='3',LPSIM_JUNCTION_EXCLUSIVE='0')
configs=a/'configs';configs.mkdir(exist_ok=True)
if args.phase=='characterize':
 ids=sorted(set(map(int,(r/'data/networks/berkeley_junction_study/regions.txt').read_text().split())))
 choices=[(f'region_{i}',[i]) for i in ids]+[('all',ids)]+[(f'union_{i}',[x for x in ids if x%4==i]) for i in range(4)]
 (configs/'characterize.txt').write_text(f'{208*1024} {len(choices)}\n'+''.join(f'{label} {len(v)} '+ ' '.join(map(str,v))+'\n' for label,v in choices))
subprocess.run(['sudo','-n','jetson_clocks','--store',str(clock)],check=True)
try:
 subprocess.run(['sudo','-n','jetson_clocks'],check=True);(a/name).mkdir(exist_ok=True)
 for batch in range(3):
  for c in json.loads((b/'prepared.json').read_text())['cases']:
   if args.phase=='evaluate' and c['pattern']!='bursty':continue
   for t in [60,180,300,450]:
    plans=[('characterize',configs/'characterize.txt')] if args.phase=='characterize' else [(f'{budget}_{scope}',configs/f"{c['name']}_{t}_{budget}_{scope}.txt") for budget in [4096,8192,16384] for scope in ['all','heldout']]
    for suffix,plan in plans:
     stem=f"{c['name']}_{t}_{batch}"+(f'_{suffix}' if args.phase=='evaluate' else '')
     subprocess.run([str(a/'index_replay'),str(r/'data/networks/berkeley_junction_study/model.txt'),str(b/'runs'/c['name']/f'snapshot_{t}.bin'),str(r/'data/networks/berkeley_junction_study/regions.txt'),str(a/name/f'{stem}.csv'),'0',str(plan)],check=True,env=e)
   print(batch,c['name'],flush=True)
finally:subprocess.run(['sudo','-n','jetson_clocks','--restore',str(clock)],check=True)
