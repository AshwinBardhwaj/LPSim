from pathlib import Path
import json,modal
image=None
if modal.is_local():
    here=Path(__file__).resolve().parent;root=here.parents[2]
    lock=json.loads((here.parent/'container-lock.json').read_text())
    image=(modal.Image.from_registry(lock['cuda_linux_amd64'],add_python='3.11')
        .add_local_dir(root/'artifacts/multigpu-preparation-20261027/source','/workspace/source',copy=True)
        .pip_install('numpy==1.26.4','pymetis==2023.1.1')
        .add_local_file(here/'local.cu','/workspace/study/local.cu',copy=True)
        .add_local_file(here/'trace.cuh','/workspace/study/trace.cuh',copy=True)
        .run_commands('mkdir -p /workspace/bin',
            'nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 -Xcompiler -pthread /workspace/study/local.cu -o /workspace/bin/local',
            'nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 -Xcompiler -pthread -DEXCHANGE_DIAGNOSTICS /workspace/study/local.cu -o /workspace/bin/diagnostic')
        .add_local_dir(here,'/workspace/study')
        .add_local_dir(root/'artifacts/prior-study-20261103/inputs','/workspace/inputs'))
app=modal.App('lpsim-prior-study')
volume=modal.Volume.from_name('lpsim-multigpu-artifacts',create_if_missing=True)

@app.function(image=image,gpu='A100-80GB:3',cpu=(4,4),memory=(8192,8192),
              timeout=1900,startup_timeout=120,retries=0,max_containers=1,
              buffer_containers=0,scaledown_window=2,single_use_containers=True,
              volumes={'/artifacts':volume})
def study():
    import subprocess,time,datetime,tarfile,hashlib,traceback,shutil
    start=time.monotonic();out=Path('/artifacts')/('prior-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    status=dict(passing=False);telemetry=None;telemetry_file=None
    def run(cmd,log,timeout,required=True):
        with (out/log).open('w') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        if required and p.returncode:raise RuntimeError(log+' failed')
    try:
        for cmd,name in [(['nvidia-smi','-q'],'hardware.txt'),(['nvidia-smi','topo','-m'],'topology.txt'),(['nvidia-smi','nvlink','--status'],'nvlink.txt')]:run(cmd,name,15,False)
        telemetry_file=(out/'telemetry.csv').open('w')
        telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv','--loop-ms=1000'],stdout=telemetry_file,stderr=subprocess.STDOUT)
        shutil.copytree('/workspace/study',out/'source',ignore=shutil.ignore_patterns('__pycache__'))
        for f in ('cases.json','sha256.json'):shutil.copy2(Path('/workspace/inputs')/f,out/('input-'+f))
        (out/'binaries.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/workspace/bin').iterdir()},indent=2))
        run(['python','/workspace/study/run.py',str(out)],'matrix.log',1810)
        status['passing']=True
    except Exception:status['error']=traceback.format_exc()
    finally:
        if telemetry:
            telemetry.terminate()
            try:telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
        if telemetry_file:telemetry_file.close()
        status['elapsed_s']=time.monotonic()-start
        (out/'status.json').write_text(json.dumps(status,indent=2))
        (out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2))
        archive=out.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz',compresslevel=1) as f:f.add(out,arcname=out.name)
        volume.commit()
    return dict(status=status,filename=archive.name,archive=archive.read_bytes())

@app.local_entrypoint()
def main(output: str):
    p=Path(output);p.mkdir(parents=True,exist_ok=False)
    result=study.remote();(p/result['filename']).write_bytes(result['archive'])
    (p/'status.json').write_text(json.dumps(result['status'],indent=2))
    print(json.dumps(result['status'],indent=2),flush=True)
    if not result['status']['passing']:raise RuntimeError('Study failed; partial artifacts preserved')
