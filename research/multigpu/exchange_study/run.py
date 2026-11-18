"""Budgeted common-backend study; training timings never include test samples."""
from pathlib import Path
import hashlib, json, os, random, statistics, subprocess, sys, time

out=Path(sys.argv[1]); inputs=Path('/workspace/inputs'); started=time.monotonic()
records=[]; counter=0; expected={}
cases=json.loads((inputs/'cases.json').read_text())
policies=['geography','paper_balanced','paper_community','work_balanced','boundary_aware']

def run(case,policy='paper_balanced',ranks=3,mode=0,phase='train',repeat=0,
        verify=False,diagnostic=False,steps=60,backend='new'):
    global counter
    if time.monotonic()-started>730: raise RuntimeError('Experiment deadline; preserve partial results')
    counter+=1; name=f'{counter:04d}_{case["name"]}_{policy}_k{ranks}_m{mode}_{phase}_r{repeat}'
    folder=out/name;folder.mkdir()
    snapshot=inputs/case['name']/'snapshot.bin';model=inputs/'models'/case['network']/'model.txt'
    owners=inputs/case['name']/f'{policy}_{ranks}.txt'
    exe='/workspace/bin/diagnostic' if diagnostic else '/workspace/bin/'+{'new':'local','legacy':'legacy','dense':'dense'}[backend]
    if backend=='dense':
        cmd=[exe,str(model),str(snapshot),str(owners),str(steps),str(folder),'single',str(int(verify)),str(int(diagnostic)),str(repeat)]
    else:
        cmd=[exe,str(model),str(snapshot),str(owners),str(steps),str(folder),str(ranks),str(int(verify)),str(int(diagnostic))]
    start=time.monotonic()
    with (folder/'run.log').open('w') as log:
        result=subprocess.run(cmd,env=dict(os.environ,LPSIM_EXCHANGE_MODE=str(mode)),stdout=log,stderr=subprocess.STDOUT,timeout=90)
    if result.returncode:raise RuntimeError(name+'\n'+(folder/'run.log').read_text())
    digest=hashlib.sha256((folder/'final.bin').read_bytes()).hexdigest()
    key=(case['name'],steps)
    if key in expected and expected[key]!=digest:raise RuntimeError('Physical hash changed: '+name)
    expected[key]=digest
    measured=json.loads((folder/'metrics.json').read_text())
    row=dict(measured)
    row.update(case=case['name'],n=case['n'],policy=policy,ranks=ranks,mode=mode,
             phase=phase,repeat=repeat,verify=verify,diagnostic=diagnostic,steps=steps,
             backend=backend,path=name,final_sha256=digest,
             process_seconds=time.monotonic()-start)
    row['compute_ms']=measured.get('compute_ms',measured.get('compute_wall_ms'))
    records.append(row);(out/'runs.json').write_text(json.dumps(records,indent=2))
    # Preserve hashes, exact-step checks and all tables, not duplicate 40-MB states.
    (folder/'final.bin').unlink()
    print(name,round(row['compute_ms'],3),flush=True)
    return row

# Physical-device correctness and independent component diagnostics first.
for case in cases:
    run(case,policy='geography',ranks=1,backend='dense',phase='gate',verify=True,steps=20)
    run(case,backend='legacy',phase='gate',verify=True,steps=20)
    for mode in range(4):
        run(case,mode=mode,phase='diagnostic',verify=True,diagnostic=True,steps=20)
    for mode in range(4):
        run(case,mode=mode,phase='backend_warmup')
    for repeat in range(3):
        order=list(range(4));random.Random(2910+repeat).shuffle(order)
        for mode in order:run(case,mode=mode,phase='backend_train',repeat=repeat)

def median(rows):return statistics.median(r['compute_ms'] for r in rows)
scores={}
for mode in range(4):
    ratios=[]
    for case in cases:
        base=median([r for r in records if r['case']==case['name'] and r['phase']=='backend_train' and r['mode']==0])
        candidate=median([r for r in records if r['case']==case['name'] and r['phase']=='backend_train' and r['mode']==mode])
        ratios.append(candidate/base)
    scores[mode]=dict(mean_ratio=statistics.mean(ratios),worst_ratio=max(ratios))
eligible=[m for m in range(1,4) if scores[m]['mean_ratio']<=.95 and scores[m]['worst_ratio']<=1.05]
chosen=min(eligible,key=lambda m:(scores[m]['mean_ratio'],m)) if eligible else 0
(out/'backend-selection.json').write_text(json.dumps(dict(chosen=chosen,scores=scores,
    rule='training mean >=5% gain and no case >5% slower; otherwise baseline',
    test_samples_used=False),indent=2))

# Every partition policy executes precisely the same chosen backend.
for case in cases:
    configs=[(p,k,chosen,'new') for k in (2,3) for p in policies]
    configs += [('paper_balanced',3,m,'new') for m in range(4) if m!=chosen]
    configs += [('geography',1,chosen,'new'),('geography',1,0,'dense'),('paper_balanced',3,0,'legacy')]
    for p,k,m,b in configs:
        if b=='new' and (p!='paper_balanced' or k!=3):
            run(case,p,k,m,'gate',verify=True,steps=20,backend=b)
        run(case,p,k,m,'policy_warmup',backend=b)
    for repeat in range(3):
        order=[c for c in configs if c[2]==chosen and c[3]=='new' and c[1]>1]
        random.Random(3920+repeat).shuffle(order)
        for p,k,m,b in order:run(case,p,k,m,'policy_train',repeat,backend=b)
    selection={}
    t=time.perf_counter()
    for k in (2,3):
        values={p:median([r for r in records if r['case']==case['name'] and r['phase']=='policy_train' and r['ranks']==k and r['policy']==p]) for p in policies}
        selection[k]=dict(policy=min(values,key=lambda p:(values[p],p)),training_medians_ms=values)
    lookup_ms=1000*(time.perf_counter()-t)
    (out/(case['name']+'-selection.json')).write_text(json.dumps(dict(
        measured_cost_candidate=selection,selection_cpu_ms=lookup_ms,test_samples_used=False,
        scope='same frozen workload, independent timing repetitions; not held-out demand'),indent=2))
    for repeat in range(5):
        order=configs[:];random.Random(4930+repeat).shuffle(order)
        for p,k,m,b in order:run(case,p,k,m,'test',repeat,backend=b)

(out/'complete.json').write_text(json.dumps(dict(passed=True,records=len(records),chosen_backend=chosen,
    exact_reference_steps=sum(r['steps'] for r in records if r['verify']),
    seconds=time.monotonic()-started,heldout_demand_test=False),indent=2))
