#!/usr/bin/env python3
"""Inspect immutable milestones and create isolated, writable research runs."""
import argparse,datetime,hashlib,json,os,re,subprocess,tarfile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ARCH=ROOT.parent/'research-archive'
def main():
 p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
 s.add_parser('list');v=s.add_parser('verify');v.add_argument('milestone',nargs='?')
 r=s.add_parser('replay');r.add_argument('milestone',choices=['san-pablo-reduction','occupancy-fix','physical-junctions','lane-preserving-following','parallel-junctions']);r.add_argument('--case',default='network_20')
 n=s.add_parser('new-run');n.add_argument('name');n.add_argument('--commit',default='HEAD')
 args=p.parse_args()
 if args.command=='list':
  for m in json.loads((ARCH/'catalog.json').read_text())['milestones']:print(m['id'],m['status'],m['path'])
 elif args.command=='verify':
  rows=json.loads((ARCH/'inventory.json').read_text());cache={};count=0
  for r in rows:
   if args.milestone and not r['path'].startswith('milestones/'+args.milestone+'/'):continue
   path=ARCH/r['path'];st=path.stat();key=(st.st_dev,st.st_ino)
   if key not in cache:
    h=hashlib.sha256()
    with path.open('rb') as f:
     for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    cache[key]=h.hexdigest()
   if cache[key]!=r['sha256'] or st.st_size!=r['bytes']:raise SystemExit('Integrity mismatch: '+r['path'])
   count+=1
  if not count:raise SystemExit('No matching archived files')
  print('Verified',count,'files')
 elif args.command=='replay':
  replay(args.milestone,args.case)
 else:
  if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*',args.name):raise SystemExit('Use letters, numbers, dashes and underscores for the run name')
  commit=subprocess.check_output(['git','rev-parse',args.commit+'^{commit}'],cwd=ROOT,text=True).strip()
  folder=ROOT/'artifacts/current'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+args.name)
  folder.mkdir(parents=True,exist_ok=False);workspace=folder/'workspace';workspace.mkdir()
  proc=subprocess.Popen(['git','archive',commit],cwd=ROOT,stdout=subprocess.PIPE)
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
   for member in tar:
    if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():raise RuntimeError('Unsafe archive member: '+member.name)
    tar.extract(member,workspace)
  if proc.wait():raise RuntimeError('git archive failed')
  (folder/'run.json').write_text(json.dumps(dict(name=args.name,commit=commit,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='prepared-not-run',workspace=str(workspace)),indent=2)+'\n')
  (folder/'README.md').write_text('This is a writable source snapshot for a new run. Build and run from workspace/.\nHistorical milestone packages are read-only and must not be used as output directories.\nRecord commands, inputs, seeds, hardware settings, diagnostics and report outputs here.\nPromote only after validation, with a new milestone ID; do not replace an existing milestone.\n')
  print(workspace)
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def replay(ident,case):
 if not re.fullmatch(r'[a-zA-Z0-9_-]+',case):raise SystemExit('Invalid case name')
 mapping={'san-pablo-reduction':'san_pablo_university_20261004','occupancy-fix':'occupancy_fix_20261004','physical-junctions':'junction_behavior_20261006','lane-preserving-following':'junction_platoons_20261007','parallel-junctions':'junction_parallel_20261007'}
 base=ARCH/'milestones'/ident/'artifacts'/mapping[ident]
 source=base/case
 if ident=='parallel-junctions':source=base/'exclusive_0'/case/'repeat_0/parallel'
 binary=base/'lpsim'
 if ident=='physical-junctions':binary=ARCH/'milestones/lane-preserving-following/artifacts/junction_platoons_20261007/source_before/lpsim'
 assert binary.is_file() and source.is_dir(),(binary,source)
 run=ROOT/'artifacts/current'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-replay-'+ident+'-'+case);run.mkdir(parents=True,exist_ok=False)
 env=dict(os.environ)
 for key in list(env):
  if key.startswith('LPSIM_'):del env[key]
 inputs=next((ARCH/'shared-inputs').glob('checkout-*'))/'data/networks'
 if ident in ('san-pablo-reduction','occupancy-fix'):
  config=json.loads((source/'case.json').read_text());network=run/'network';shutil.copytree(inputs/config['network'],network)
  shutil.copy2(source/'demand.csv',network/'demand.csv');(run/'data').mkdir()
  text=(source/'data/command_line_options.ini').read_text();text=re.sub(r'^NETWORK_PATH=.*$',f'NETWORK_PATH={network}/',text,flags=re.M);text=re.sub(r'^OD_DEMAND_FILENAME=.*$','OD_DEMAND_FILENAME=demand.csv',text,flags=re.M);(run/'data/command_line_options.ini').write_text(text)
  env.update(LPSIM_SIGNAL_MODE=config['signal_mode'],LPSIM_SIGNALS=str(run/'signals.csv'),LPSIM_TRAJECTORIES=str(run/'trajectories.csv'),LPSIM_METRICS=str(run/'metrics.json'),OMP_NUM_THREADS='2')
 else:
  scenario=source/'scenario.txt'
  if ident=='parallel-junctions':scenario=ARCH/'milestones/lane-preserving-following/artifacts/junction_platoons_20261007'/case/'scenario.txt'
  lines=scenario.read_text().splitlines();model=inputs/Path(lines[0]).parent.name/'model.txt';assert model.exists();shutil.copy2(model,run/'model.txt');lines[0]=str(run/'model.txt');(run/'scenario.txt').write_text('\n'.join(lines)+'\n')
  env.update(LPSIM_JUNCTION_SCENARIO=str(run/'scenario.txt'),LPSIM_JUNCTION_EXCLUSIVE='0',LPSIM_JUNCTION_PARALLEL='1' if ident=='parallel-junctions' else '0')
 command=[str(binary)]
 (run/'command.json').write_text(json.dumps(dict(command=command,env={k:v for k,v in env.items() if k.startswith('LPSIM_') or k=='OMP_NUM_THREADS'},binary_sha256=digest(binary)),indent=2))
 with (run/'run.log').open('w') as log:r=subprocess.run(command,cwd=run,env=env,stdout=log,stderr=subprocess.STDOUT)
 if r.returncode:raise SystemExit('Replay failed: '+str(run/'run.log'))
 expected=json.loads((source/'metrics.json').read_text());actual=json.loads((run/'metrics.json').read_text());checks={}
 for key in ('completed','active','pending','steps','gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples','entries','right_on_red_entries'):
  if key in expected:checks[key]=dict(expected=expected[key],actual=actual[key],match=expected[key]==actual[key])
 exact={name:digest(source/name)==digest(run/name) for name in ('states.csv','events.csv','signals.csv','trajectories.csv') if (source/name).exists() and (run/name).exists()}
 passed=bool(checks) and all(x['match'] for x in checks.values()) and all(exact.values());result=dict(milestone=ident,case=case,passed=passed,behavioral_metrics=checks,byte_identical_files=exact,original=str(source),output=str(run),timing_compared=False)
 (run/'replay-validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if not passed:raise SystemExit('Behavioral replay mismatch')

if __name__=='__main__':main()
