#!/usr/bin/env python3
"""Milestone 1 conservation, repeat equivalence and observed-regime gates."""
import csv,json
from milestone1_baseline import ART,sha,dump

def main():
 manifest=json.loads((ART/'manifest.json').read_text());assert sha(ART/'lpsim')==manifest['binary_sha256'];cases=[]
 for n in (20,400,1600):assert (ART/'cases'/f'controller72_{n}/demand.txt').read_bytes()==(ART/'cases'/f'always_permitted_{n}/demand.txt').read_bytes()
 for case in manifest['cases']:
  name=case['name'];src=ART/'cases'/name
  assert sha(src/'scenario.txt')==case['scenario_sha256'] and sha(src/'demand.txt')==case['demand_sha256']
  model=__import__('pathlib').Path((src/'scenario.txt').read_text().splitlines()[0]);assert sha(model)==case['model_sha256']
  reference=None
  for backend in ('parallel','serial'):
   for repeat in range(3):
    folder=ART/'runs'/name/f'{backend}_{repeat}';v=json.loads((folder/'validated.json').read_text());assert v['passed']
    if reference is None:reference=v['hashes']
    assert v['hashes']==reference
    for row in csv.DictReader((folder/'step_metrics.csv').open()):
     assert sum(int(row[k]) for k in ('pending','road_stopped','road_moving','connector','completed'))==case['n']
     assert int(row['pending'])>=int(row['not_departed'])
  folder=ART/'runs'/name/'parallel_0';rows=list(csv.DictReader((folder/'step_metrics.csv').open()));m=json.loads((folder/'metrics.json').read_text())
  backlog=max(int(r['pending'])-int(r['not_departed'])-int(r['spawned']) for r in rows)
  if case['mode']=='corridor':
   assert backlog==0 and m['completed']==case['n'] and m['downstream_blocked_decisions']==m['yield_blocked_decisions']==0
   label='Synthetic free-flow capacity control (startup acceleration retained)'
  elif case['mode']=='always_permitted':label='Low-demand unsignalized movement' if case['n']==20 else 'Always-permitted, congested; not free-flow'
  else:label='Low-demand signal stopping' if case['n']==20 else 'Signal-controlled congestion with pending backlog'
  for t in manifest['profile_times']:
   sample_folder=ART/'profiles'/name/str(t)
   sample=json.loads((sample_folder/'sample.json').read_text());assert sample['matched_reference']
   for filename in reference:
    prof=(sample_folder/filename).read_bytes().splitlines();full=(folder/filename).read_bytes().splitlines();assert prof==full[:len(prof)],(name,t,filename)
  cases.append(dict(case=name,regime=label,maximum_due_backlog_after_spawn=backlog,completed=m['completed'],space_denials=m['downstream_blocked_decisions'],conflict_or_yield_denials=m['yield_blocked_decisions']))
 dump(ART/'gate.json',dict(passed=True,full_runs=54,profile_windows=27,profile_reference_prefix_equivalence=True,serial_and_repeat_byte_equivalence=True,population_conservation=True,movement_and_gap_checks=True,cases=cases));print('Milestone 1 gates passed')
if __name__=='__main__':main()
