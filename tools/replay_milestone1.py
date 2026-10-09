#!/usr/bin/env python3
"""Replay a preserved baseline case into a fresh directory on a compatible Jetson."""
import argparse,datetime,json,os,subprocess
from milestone1_baseline import ART,ROOT,sha,dump

def main():
 p=argparse.ArgumentParser();p.add_argument('--case',default='controller72_20');p.add_argument('--serial',action='store_true');a=p.parse_args()
 manifest=json.loads((ART/'manifest.json').read_text());case=next((c for c in manifest['cases'] if c['name']==a.case),None)
 if case is None:raise SystemExit('Unknown case')
 assert sha(ART/'lpsim')==manifest['binary_sha256']
 folder=ROOT/'artifacts/current'/(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-replay-milestone1-'+a.case);folder.mkdir(parents=True,exist_ok=False)
 lines=(ART/'cases'/a.case/'scenario.txt').read_text().splitlines();model=ART/'inputs'/('corridor.txt' if case['mode']=='corridor' else 'san_pablo.txt');assert sha(model)==case['model_sha256'];lines[0]=str(model);(folder/'scenario.txt').write_text('\n'.join(lines)+'\n')
 env={k:v for k,v in os.environ.items() if not k.startswith('LPSIM_')};env.update(LPSIM_JUNCTION_SCENARIO=str(folder/'scenario.txt'),LPSIM_JUNCTION_PARALLEL='0' if a.serial else '1',LPSIM_JUNCTION_EXCLUSIVE='0')
 with (folder/'run.log').open('w') as log:subprocess.run([str(ART/'lpsim')],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 original=json.loads((ART/'runs'/a.case/'parallel_0/validated.json').read_text());checks={name:sha(folder/name)==value for name,value in original['hashes'].items()}
 dump(folder/'validation.json',dict(case=a.case,passed=all(checks.values()),byte_identical=checks,original_commit=manifest['commit']))
 assert all(checks.values());print(folder)
if __name__=='__main__':main()
