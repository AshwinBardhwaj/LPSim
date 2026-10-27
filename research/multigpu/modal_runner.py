"""Prepared Modal runner. Importing does not launch GPUs; `modal run` does.

LPSIM_BUNDLE=/absolute/frozen/bundle modal run research/multigpu/modal_runner.py
"""
from pathlib import Path
import json, os
import modal

HERE=Path(__file__).resolve().parent
LOCK=json.loads((HERE/'container-lock.json').read_text())
BUNDLE=Path(os.environ.get('LPSIM_BUNDLE',str(HERE.parents[1]/'artifacts/multigpu-preparation-20261027'))).resolve()
# Include only frozen source and inputs, not the user's workspace or credentials.
image=(modal.Image.from_registry(LOCK['cuda_linux_amd64'],add_python=LOCK['python'])
       .add_local_dir(BUNDLE/'source','/workspace/source',copy=True)
       .run_commands('bash /workspace/source/research/multigpu/build.sh /workspace/bin sm_80')
       .add_local_dir(BUNDLE/'inputs','/workspace/inputs')
       .add_local_file(BUNDLE/'freeze.json','/workspace/freeze.json')
       .add_local_file(BUNDLE/'workloads.json','/workspace/workloads.json'))
app=modal.App('lpsim-three-partition-preflight')
volume=modal.Volume.from_name('lpsim-multigpu-artifacts',create_if_missing=True)

@app.function(image=image,gpu=LOCK['gpu_request'],cpu=4,memory=16384,timeout=1800,volumes={'/artifacts':volume})
def preflight(case: str='reference_8192_t180',policy: str='geography',steps: int=120):
    import datetime, hashlib, platform, shutil, subprocess, tarfile, traceback, uuid
    root=Path('/workspace'); inputs=root/'inputs'; manifest=json.loads((root/'workloads.json').read_text())
    known={c['name'] for c in manifest['cases']}
    if case not in known or policy not in ('geography','vehicle_count','communication_graph') or not 1<=steps<=2400:
        raise ValueError('Unknown case/policy or steps outside 1..2400')
    run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=Path('/artifacts')/run_id;out.mkdir();commands=[]
    def run(cmd,name,required=False):
        with (out/name).open('w') as log:
            try: p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,text=True,timeout=900);code=p.returncode
            except (FileNotFoundError,subprocess.TimeoutExpired) as e:log.write(str(e));code=-1
        commands.append(dict(argv=cmd,output=name,exit_code=code))
        if required and code:raise RuntimeError(f'{name} failed, see saved artifacts')
        return code
    status={'run_id':run_id,'purpose':'correctness/hardware preflight; not concurrent scaling benchmark','case':case,'policy':policy,'steps':steps,'platform':platform.platform()}
    try:
        # Validate frozen inputs/source before using them; binary hashes after x86 build.
        run(['python','/workspace/source/research/multigpu/verify_bundle.py',str(root)],'hash-verification.log',True)
        for cmd,name in [(['nvidia-smi','-q'],'nvidia-smi.txt'),(['nvidia-smi','topo','-m'],'topology.txt'),(['nvidia-smi','nvlink','--status'],'nvlink.txt'),(['ncu','--version'],'ncu-version.txt'),(['nsys','--version'],'nsys-version.txt')]:run(cmd,name)
        run([str(root/'bin/hardware_probe')],'hardware.json',True)
        hardware=json.loads((out/'hardware.json').read_text())
        if len(hardware['devices'])!=3 or any('A100' not in d['name'] for d in hardware['devices']):raise RuntimeError('Expected exactly three A100 devices')
        status['counter_probe_exit_code']=run(['ncu','--target-processes','all','--set','basic','--launch-count','1','--export',str(out/'counter-probe'),str(root/'bin/hardware_probe'),'counter'],'counter-probe.log')
        for layout,devices in [('logical','0,0,0'),('physical','0,1,2')]:
            d=out/layout;d.mkdir();cmd=[str(root/'bin/partitioned'),str(inputs/'network/model.txt'),str(inputs/case/'snapshot.bin'),str(inputs/case/(policy+'.txt')),str(steps),str(d),'bulk','0',devices]
            run(cmd,layout+'.log',True)
        a=(out/'logical/final.bin').read_bytes();b=(out/'physical/final.bin').read_bytes()
        if a!=b:raise RuntimeError('Logical/physical final states differ')
        status.update(correctness_pass=True,final_sha256=hashlib.sha256(a).hexdigest())
        # Optional trace is a separate run; it cannot enter benchmark timings.
        if shutil.which('nsys'):
            d=out/'trace';d.mkdir();run(['nsys','profile','--trace=cuda,osrt','--force-overwrite=true','-o',str(out/'partition-trace'),str(root/'bin/partitioned'),str(inputs/'network/model.txt'),str(inputs/case/'snapshot.bin'),str(inputs/case/(policy+'.txt')),'4',str(d),'bulk','0','0,1,2'],'nsys.log')
    except Exception:
        status.update(correctness_pass=False,error=traceback.format_exc())
    finally:
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        shutil.copy2(root/'freeze.json',out/'freeze.json');shutil.copy2(root/'workloads.json',out/'workloads.json')
        for p in (root/'bin').glob('*.txt'):shutil.copy2(p,out/p.name)
        shutil.copy2(root/'bin/binaries.sha256',out/'binaries.sha256')
        hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
        (out/'output-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
        archive=Path('/artifacts')/(run_id+'.tar.gz')
        with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=run_id)
        volume.commit()
    return dict(status=status,volume='lpsim-multigpu-artifacts',archive=archive.name)

@app.local_entrypoint()
def main(case: str='reference_8192_t180',policy: str='geography',steps: int=120):
    result=preflight.remote(case,policy,steps)
    print(json.dumps(result,indent=2))
    if not result['status'].get('correctness_pass'):raise RuntimeError('Preflight failed; artifact bundle preserved')
