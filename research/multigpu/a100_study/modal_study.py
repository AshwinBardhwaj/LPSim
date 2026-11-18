"""Bounded A100 study. Each invocation is single-use, with local result retrieval."""
from pathlib import Path
import json, modal
image=None
if modal.is_local():
    here=Path(__file__).resolve().parent
    bundle=here.parents[2]/'artifacts/multigpu-preparation-20261027'
    lock=json.loads((here.parent/'container-lock.json').read_text())
    image=(modal.Image.from_registry(lock['cuda_linux_amd64'],add_python='3.11')
           .add_local_dir(bundle/'source','/workspace/source',copy=True)
           .run_commands('mkdir -p /workspace/bin && nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 /workspace/source/research/multigpu/partitioned.cu -o /workspace/bin/partitioned')
           .add_local_dir(here,'/workspace/study',copy=True)
           .run_commands('if test -f /workspace/study/concurrent.cu; then nvcc -O3 --use_fast_math -lineinfo -std=c++17 -arch=sm_80 -Xcompiler -pthread /workspace/study/concurrent.cu -o /workspace/bin/concurrent; fi')
           .add_local_dir(bundle/'inputs','/workspace/inputs')
           .add_local_file(bundle/'workloads.json','/workspace/workloads.json'))
app=modal.App('lpsim-a100-partition-study')
volume=modal.Volume.from_name('lpsim-multigpu-artifacts',create_if_missing=True)
@app.function(image=image,gpu='A100-80GB:3',cpu=(4,4),memory=(8192,8192),timeout=1100,startup_timeout=120,retries=0,max_containers=1,buffer_containers=0,scaledown_window=2,single_use_containers=True,volumes={'/artifacts':volume})
def study(phase: str='gate'):
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
        shutil.copy2('/workspace/workloads.json',out/'workloads.json')
        if phase=='gate':
            case='reference_8192_t180'
            for policy in ['geography','vehicle_count','communication_graph']:
                results=[]
                for name,devices in [('logical','0,0,0'),('physical','0,1,2')]:
                    d=out/(policy+'-'+name);d.mkdir()
                    run(['/workspace/bin/partitioned','/workspace/inputs/network/model.txt',f'/workspace/inputs/{case}/snapshot.bin',f'/workspace/inputs/{case}/{policy}.txt','120',str(d),'bulk','0',devices],policy+'-'+name+'.log',110)
                    results.append((d/'final.bin').read_bytes())
                if results[0]!=results[1]:raise RuntimeError('Logical/physical mismatch: '+policy)
        elif phase=='profile':
            run(['python','/workspace/study/profile.py',str(out)],'profiles.log',900)
        else:
            run(['python','/workspace/study/run_matrix.py',str(out),phase],'matrix.log',950)
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
def main(output: str,phase: str='gate'):
    p=Path(output);p.mkdir(parents=True,exist_ok=False)
    result=study.remote(phase);(p/result['filename']).write_bytes(result['archive']);(p/'status.json').write_text(json.dumps(result['status'],indent=2));print(json.dumps(result['status'],indent=2),flush=True)
    if not result['status']['passing']:raise RuntimeError('Study failed; partial results preserved')
