#!/usr/bin/env python3
import csv,json,math
from collections import defaultdict
from milestone2_snapshots import ART,dump,sha

def rows(path):return list(csv.DictReader(path.open()))
def main():
 cfg=json.loads((ART/'manifest.json').read_text());model=(ART/'model.txt').read_text().splitlines();nr=int(model[0].split()[0]);roads=[list(map(float,l.split())) for l in model[1:1+nr]];reference=None;checks=[]
 for case in cfg['cases']:
  name=case['name'];raw=(ART/'snapshots'/f'{name}.txt').read_text().splitlines();assert sha(ART/'snapshots'/f'{name}.txt')==case['sha256']
  before={int(r[0]):r for r in (line.split() for line in raw[1:])};active={i:r for i,r in before.items() if int(r[1])==1};assert len(active)==case['active'] and len(before)==case['total']
  for phase,records in [('input',[dict(id=i,state=r[1],edge=r[2],pos=r[3],speed=r[4]) for i,r in before.items()]),('output',rows(ART/'runs'/name/'0.csv'))]:
   lanes=defaultdict(list)
   for r in records:
    if int(r['state'])!=1:continue
    edge=int(r['edge']);pos=float(r['pos']);speed=float(r['speed']);assert math.isfinite(pos) and math.isfinite(speed) and 0<=pos<=roads[edge][3] and speed>=0
    lanes[edge].append(pos)
   for positions in lanes.values():
    positions.sort();assert all(b-a>=6.5-1e-5 for a,b in zip(positions,positions[1:])),(name,phase)
  baseline=ART/'runs'/name/'0.csv'
  for rep in (1,2):assert sha(baseline)==sha(ART/'runs'/name/f'{rep}.csv')
  assert sha(baseline)==sha(ART/'runs'/name/'instrumented.csv')
  output={int(r['id']):r for r in rows(baseline) if int(r['id']) in active}
  if name=='motion_50':reference=output
  if case['group'] in ('padding','order'):assert output==reference,name
  count=json.loads((ART/'runs'/name/'instrumented.csv.counts.json').read_text());n=case['total'];a=case['active'];assert count==dict(road_updates=a,receiving_candidates=a*n,leader_candidates=a*(n-1),connector_candidates=a*n,stopline_candidates=0)
  profile=ART/'profiles'/name/'output.csv';assert sha(profile)==sha(baseline)
  checks.append(dict(case=name,spacing_passed=True,repeat_and_profile_identical=True,search_counts_verified=True,active_physical_equivalence=case['group'] in ('padding','order')))
 dump(ART/'validation.json',dict(passed=True,cases=checks,scope='Road kernel only; no intersection requests executed; stable IDs used to compare physical output after storage permutation.'))
 print('All snapshot validation checks passed')
if __name__=='__main__':main()
