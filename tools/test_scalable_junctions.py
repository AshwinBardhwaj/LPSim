from pathlib import Path
import subprocess,os,hashlib,json
root=Path.cwd();art=root/'artifacts/current/berkeley-state-study-20261014';src=root/'artifacts/junction_platoons_20261007'
results=[]
for case in sorted(src.glob('*/case.json')):
 name=case.parent.name
 for variant in ('reference','serial','parallel','indexed'):
  folder=art/'validation'/name/variant;folder.mkdir(parents=True,exist_ok=True)
  env=dict(os.environ,LPSIM_JUNCTION_SCENARIO=str(case.parent/'scenario.txt'),LPSIM_JUNCTION_PARALLEL='0' if variant=='serial' else '1',LPSIM_LEADER_INDEX='1' if variant=='indexed' else '0')
  cmd=[str(root/'artifacts/current/milestone1-baseline-20261008/lpsim')] if variant=='reference' else [str(art/'simulator'),str(case.parent/'scenario.txt')]
  with (folder/'run.log').open('w') as log:subprocess.run(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  if variant!='reference':
   for file in ('states.csv','events.csv','timeseries.csv','signals.csv'):
    assert hashlib.sha256((folder/file).read_bytes()).digest()==hashlib.sha256((folder.parent/'reference'/file).read_bytes()).digest(),(name,variant,file)
 results.append(name);print(name,'four-way exact parity',flush=True)
(art/'validation.json').write_text(json.dumps({'passed':True,'cases':results,'comparison':'frozen Milestone1 binary, scalable serial, scalable parallel, indexed parallel; exact physical CSV parity'},indent=2))
