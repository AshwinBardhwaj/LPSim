"""Bounded A100 study. Each invocation is single-use, with local result retrieval."""
from pathlib import Path
import json, modal
image=None
if modal.is_local():
    here=Path(__file__).resolve().parent
    bundle=here.parents[2]/'artifacts/multigpu-preparation-20261027'
    lock=json.loads((here.parent/'container-lock.json').read_text())
    data=here.parents[2]/'artifacts/local-scaling-20261028/inputs'
    image=(modal.Image.from_registry(lock['cuda_linux_amd64'],add_python='3.11')
           .add_local_dir(bundle/'source','/workspace/source',copy=True)
           .add_local_dir(here,'/workspace/study',copy=True)
           .add_local_file(here.parent/'a100_study/concurrent.cu','/workspace/dense/concurrent.cu',copy=True)
           .run_commands('mkdir -p /workspace/bin && nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 -Xcompiler -pthread /workspace/study/local.cu -o /workspace/bin/local && nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 -Xcompiler -pthread /workspace/dense/concurrent.cu -o /workspace/bin/dense')
           .add_local_dir(data,'/workspace/inputs'))
app=modal.App('lpsim-local-scaling')
volume=modal.Volume.from_name('lpsim-multigpu-artifacts',create_if_missing=True)
@app.function(image=image,gpu='A100-80GB:3',cpu=(4,4),memory=(8192,8192),timeout=1000,startup_timeout=120,retries=0,max_containers=1,buffer_containers=0,scaledown_window=2,single_use_containers=True,volumes={'/artifacts':volume})
def study(phase: str='scaling'):
    import subprocess,time,datetime,tarfile,hashlib,traceback,shutil
    started=time.monotonic();out=Path('/artifacts')/('study-'+phase+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir()
    status={'phase':phase,'passing':False};commands=[];telemetry=None;telemetry_file=None
    def run(cmd,log,limit=180,required=True):
        t=time.monotonic()
        with (out/log).open('w') as f:
            try:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=limit);code=r.returncode
            except Exception as e:f.write(str(e));code=-1
        commands.append(dict(argv=cmd,exit_code=code,seconds=time.monotonic()-t,log=log));print(log,code,round(time.monotonic()-t,2),flush=True)
        if code and required:raise RuntimeError(log+' failed')
    try:
        run(['nvidia-smi','-q'],'hardware.txt',15)
        run(['nvidia-smi','topo','-m'],'topology.txt',15,False)
        run(['nvidia-smi','nvlink','--status'],'nvlink-status.txt',15,False)
        telemetry_file=(out/'telemetry.csv').open('w');telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv','--loop-ms=1000'],stdout=telemetry_file,stderr=subprocess.STDOUT)
        run(['bash','-c','nvcc --version; find /workspace/bin /workspace/study -type f -exec sha256sum {} +'],'build.txt',15)
        shutil.copytree('/workspace/study',out/'study-source',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copy2('/workspace/inputs/cases.json',out/'cases.json')
        run(['python','/workspace/study/run.py',str(out)],'matrix.log',850)
        status['passing']=True
    except Exception:status['error']=traceback.format_exc()
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
        if telemetry_file:telemetry_file.close()
        status['elapsed_s']=time.monotonic()-started
        (out/'status.json').write_text(json.dumps(status,indent=2));(out/'commands.json').write_text(json.dumps(commands,indent=2))
        (out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2))
        archive=out.with_suffix('.tar.gz')
        with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
        volume.commit()
    return dict(status=status,filename=archive.name,archive=archive.read_bytes())
@app.local_entrypoint()
def main(output: str,phase: str='scaling'):
    p=Path(output);p.mkdir(parents=True,exist_ok=False)
    result=study.remote(phase);(p/result['filename']).write_bytes(result['archive']);(p/'status.json').write_text(json.dumps(result['status'],indent=2));print(json.dumps(result['status'],indent=2),flush=True)
    if not result['status']['passing']:raise RuntimeError('Study failed; partial results preserved')
