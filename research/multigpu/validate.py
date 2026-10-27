"""Correctness suite. GPU access required; no cloud execution."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time
from prepare import ROOT, POLICIES, sha

def run(root, binary):
    root=root.resolve(); binary=binary.resolve(); outputs=root/'validation'; outputs.mkdir(exist_ok=True)
    jobs=[]
    for policy in POLICIES:
        jobs += [('reference_2048_t0',policy,2400,'bulk',0),('reference_8192_t180',policy,120,'bulk',0),('reference_16384_t180',policy,40,'bulk',0),('test_bursty_2048',policy,2400,'bulk',0)]
    jobs += [('reference_2048_t0','geography',2400,'incremental',0),('reference_8192_t180','communication_graph',120,'incremental',0)]
    jobs += [('reference_8192_t180','geography',80,'bulk',mode) for mode in [1,2]]
    rows=[]
    for case,policy,steps,index,mode in jobs:
        name=f'{case}-{policy}-{index}-mode{mode}'; out=outputs/name; out.mkdir(exist_ok=False)
        cmd=[str(binary),str(root/'inputs/network/model.txt'),str(root/'inputs'/case/'snapshot.bin'),str(root/'inputs'/case/(policy+'.txt')),str(steps),str(out),index,str(mode)]
        (out/'command.json').write_text(json.dumps(dict(argv=cmd,env={'LPSIM_MOTION_CONTEXT':'3'}),indent=2)+'\n')
        start=time.monotonic()
        with (out/'run.log').open('w') as log: proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
        if proc.returncode: raise RuntimeError(name+': '+(out/'run.log').read_text()[-2000:])
        metrics=json.loads((out/'metrics.json').read_text()); row=dict(name=name,case=case,policy=policy,index=index,mode=mode,process_s=time.monotonic()-start,metrics=metrics,final_sha256=sha(out/'final.bin'))
        if case=='reference_2048_t0':
            frozen=ROOT/'artifacts/current/compact-motion-20261022/runs/distributed_2048_seed1901'
            checks={}
            for snap in sorted(out.glob('snapshot_*.bin')):
                prior=frozen/snap.name
                if prior.exists():
                    checks[snap.name]=sha(prior)==sha(snap)
                    if not checks[snap.name]: raise RuntimeError('Frozen baseline checkpoint mismatch '+str(snap))
            row['frozen_checkpoint_checks']=checks
        rows.append(row);(root/'validation_results.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(name,'PASS',metrics['steps'],'steps','migrations',metrics['ownership_transfers'],flush=True)
    # Baseline choice must not change final state between policies or index modes.
    groups={}
    for row in rows:
        key=(row['case'],row['mode'],row['metrics']['steps'])
        groups.setdefault(key,set()).add(row['final_sha256'])
    if any(len(x)!=1 for x in groups.values()): raise RuntimeError('Policy-dependent physical results')
    # Fail closed when a message pool is too small; never truncate silently.
    out=outputs/'capacity_failure';out.mkdir(exist_ok=False)
    case=root/'inputs/reference_8192_t180'
    cmd=[str(binary),str(root/'inputs/network/model.txt'),str(case/'snapshot.bin'),str(case/'geography.txt'),'1',str(out),'bulk','0','0,0,0','1']
    p=subprocess.run(cmd,text=True,capture_output=True);(out/'run.log').write_text(p.stdout+p.stderr)
    if p.returncode==0 or 'capacity exceeded' not in p.stderr: raise RuntimeError('Capacity guard did not fail closed')
    (root/'validation_gate.json').write_text(json.dumps(dict(exact=True,physical_invariants=True,policy_independent=True,capacity_guard=True,runs=len(rows),compared_timesteps=sum(r['metrics']['steps'] for r in rows)),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('binary',type=Path);a=p.parse_args();run(a.root,a.binary)
