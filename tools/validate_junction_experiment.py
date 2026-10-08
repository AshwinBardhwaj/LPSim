#!/usr/bin/env python3
"""Independent generated lane-rule and matched-run admission audits."""
import csv,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_platoons_20261007';NET=ROOT/'data/networks/san_pablo_lane_preserving'
def main():
 model=json.loads((NET/'junction_model.json').read_text());counts={'left':0,'right':0,'through':0}
 for m in model['movements']:
  a,b=model['roads'][m['incoming']],model['roads'][m['outgoing']]
  if m['turn']==0:assert m['in_lane']==m['out_lane'];counts['through']+=1
  if m['turn']==1:assert m['in_lane']==m['out_lane']==0;counts['left']+=1
  if m['turn']==-1:assert m['in_lane']==a['lanes']-1 and m['out_lane']==b['lanes']-1;counts['right']+=1
 audited=[]
 for folder in list(ART.glob('*/validation.json'))+list((ART/'exclusive').glob('*/validation.json')):
  folder=folder.parent;m=json.loads((folder/'metrics.json').read_text());assert all(m[k]==0 for k in ('gap_failures','conflict_failures','invalid_states','stopped_in_junction_samples'))
  assert m['trips']==m['completed']+m['active']+m['pending'];previous={};minimum=1e9
  for e in csv.DictReader((folder/'events.csv').open()):
   mid=int(e['movement']);t=float(e['time']);movement=model['movements'][mid]
   assert int(e['turn'])==movement['turn']
   assert int(e['light'])==2 or (int(e['event'])==2 and int(e['turn'])==-1 and int(e['light'])==0 and float(e['wait_seconds'])>=2)
   if mid in previous:minimum=min(minimum,t-previous[mid]);assert t-previous[mid]>=2-1e-6
   previous[mid]=t
  audited.append(dict(case=str(folder.relative_to(ART)),minimum_same_movement_entry_headway_seconds=minimum if minimum<1e9 else None))
 for n in (20,400,1600):
  for f in ('scenario.txt','demand.json'):
   assert (ART/f'network_{n}'/f).read_bytes()==(ART/'exclusive'/f'network_{n}'/f).read_bytes()
 before=json.loads((ART/'exclusive/same_movement_platoon/metrics.json').read_text());after=json.loads((ART/'same_movement_platoon/metrics.json').read_text())
 assert before['max_same_movement_occupancy']==1 and after['max_same_movement_occupancy']>=2
 result=dict(passed=True,connector_lane_rules=counts,audited_cases=audited,matched_demand_identical=True,binary_sha256=hashlib.sha256((ROOT/'build/lpsim').read_bytes()).hexdigest())
 (ART/'independent_validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
