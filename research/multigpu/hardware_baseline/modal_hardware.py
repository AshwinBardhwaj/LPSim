"""Single, bounded three-A100 hardware baseline. No traffic simulation is run."""
from pathlib import Path
import json
import modal
image=None
if modal.is_local():
    HERE=Path(__file__).resolve().parent
    LOCK=json.loads((HERE.parent/'container-lock.json').read_text())
    image=(modal.Image.from_registry(LOCK['cuda_linux_amd64'],add_python='3.11')
           .add_local_file(HERE/'benchmark.cu','/bench/benchmark.cu',copy=True)
           .run_commands('nvcc -O3 -lineinfo -std=c++17 -arch=sm_80 /bench/benchmark.cu -lcublas -Xcompiler -pthread -o /bench/benchmark',
                         'nvcc --version > /bench/build.txt',
                         'sha256sum /bench/benchmark /bench/benchmark.cu > /bench/source-binary.sha256'))
app=modal.App('lpsim-a100-hardware-baseline')
volume=modal.Volume.from_name('lpsim-multigpu-artifacts',create_if_missing=True)

@app.function(image=image,gpu='A100-40GB:3',cpu=(4,4),memory=(8192,8192),
              timeout=600,startup_timeout=120,retries=0,max_containers=1,
              buffer_containers=0,scaledown_window=2,single_use_containers=True,
              volumes={'/artifacts':volume})
def baseline(profile_only: bool=False):
    import datetime,hashlib,os,platform,shutil,subprocess,tarfile,time,traceback,uuid
    started=time.monotonic();run_id='hardware-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    out=Path('/artifacts')/run_id;out.mkdir();commands=[]
    def run(argv,name,timeout=15,required=False):
        before=time.monotonic()
        with (out/name).open('w') as f:
            try:result=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=timeout);code=result.returncode
            except (subprocess.TimeoutExpired,FileNotFoundError) as e:f.write(str(e)+'\n');code=-1
        commands.append(dict(argv=argv,log=name,exit_code=code,wall_s=time.monotonic()-before))
        if required and code:raise RuntimeError(name+' failed')
        return code
    status=dict(run_id=run_id,requested_gpu='A100-40GB:3',cloud=platform.platform(),purpose='hardware-only baseline',gpu_timeout_s=600,startup_timeout_s=120,retries=0,maximum_containers=1,passing=False,profile_only=profile_only)
    telemetry=None;telemetry_file=None
    try:
        for argv,name in [(['nvidia-smi','-L'],'gpu-list.txt'),(['nvidia-smi','-q'],'nvidia-smi-before.txt'),(['nvidia-smi','topo','-m'],'topology.txt'),(['nvidia-smi','topo','-p2p','r'],'p2p-read.txt'),(['nvidia-smi','topo','-p2p','w'],'p2p-write.txt'),(['nvidia-smi','nvlink','--status'],'nvlink-status.txt'),(['nvidia-smi','nvlink','--capabilities'],'nvlink-capabilities.txt'),(['nvidia-smi','--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory','--format=csv'],'gpu-properties.csv'),(['lscpu'],'cpu.txt'),(['ncu','--version'],'ncu-version.txt'),(['nsys','--version'],'nsys-version.txt'),(['bash','-c','cat /proc/self/status /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max 2>/dev/null'],'container-limits.txt')]:run(argv,name)
        telemetry_file=(out/'telemetry.csv').open('w');telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,clocks.sm,clocks.mem,power.draw,temperature.gpu','--format=csv','--loop-ms=1000'],stdout=telemetry_file,stderr=subprocess.STDOUT)
        if profile_only:
            run(['/bench/benchmark',str(out),'counter'],'benchmark.log',timeout=15,required=True)
            status['passing']=True
            run(['nvidia-smi','nvlink','--remote-pci-info'],'nvlink-remote-pci.txt')
        else:
            run(['/bench/benchmark',str(out)],'benchmark.log',timeout=300,required=True)
            status['passing']=(out/'PASS').exists()
        run(['nvidia-smi','-q'],'nvidia-smi-after.txt')
        # Profile separately. Counter access may be prohibited by the host driver.
        status['ncu_exit_code']=run(['ncu','--clock-control','none','--target-processes','all','--set','basic','--launch-count','1','--export',str(out/'counter-probe'),'/bench/benchmark',str(out),'counter'],'counter-probe.log',timeout=35)
        status['nsys_exit_code']=run(['nsys','profile','--trace=cuda,osrt','--force-overwrite=true','-o',str(out/'launch-trace'),'/bench/benchmark',str(out),'counter'],'nsys-probe.log',timeout=35)
    except Exception:status['error']=traceback.format_exc()
    finally:
        if telemetry is not None:
            telemetry.terminate()
            try:telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired:telemetry.kill();telemetry.wait()
        if telemetry_file:telemetry_file.close()
        for file in ['benchmark.cu','build.txt','source-binary.sha256']:shutil.copy2('/bench/'+file,out/file)
        status['function_elapsed_s']=time.monotonic()-started
        # Rate-based estimate for the function body only; not a provider invoice.
        status['rates_usd_per_second']={'gpu_each':.000583,'cpu_core':.0000131,'memory_gib':.00000222}
        status['body_cost_estimate_usd']=status['function_elapsed_s']*(3*.000583+4*.0000131+8*.00000222)
        (out/'status.json').write_text(json.dumps(status,indent=2)+'\n');(out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        (out/'sha256.json').write_text(json.dumps({str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()},indent=2)+'\n')
        archive=Path('/artifacts')/(run_id+'.tar.gz')
        with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=run_id)
        volume.commit()
    return dict(status=status,archive=archive.read_bytes(),filename=archive.name)

@app.local_entrypoint()
def main(output: str='artifacts/modal-hardware-20261027',profile_only: bool=False):
    import datetime,hashlib
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    # One invocation only. Conservative reservation covers 600 s body + startup,
    # image-build CPU, and result handling; no automatic reruns are authorized here.
    ledger={'authorized_budget_usd':5,'reserved_upper_allowance_usd':2,'invocations':1,'gpu_runtime_limit_s':600,'startup_limit_s':120,'retries':0,'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pricing_source':'https://modal.com/pricing','pricing_checked':'2026-10-27'}
    (root/'budget.json').write_text(json.dumps(ledger,indent=2)+'\n')
    try:
        result=baseline.remote(profile_only);archive=root/result['filename'];archive.write_bytes(result['archive']);(root/'status.json').write_text(json.dumps(result['status'],indent=2)+'\n');(root/'archive.sha256').write_text(hashlib.sha256(result['archive']).hexdigest()+'  '+archive.name+'\n');print(json.dumps(result['status'],indent=2));print('Saved',archive)
        if not result['status']['passing']:raise RuntimeError('Hardware test failed; partial artifacts preserved')
    except Exception as e:
        ledger['local_error']=str(e);raise
    finally:
        ledger['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(root/'budget.json').write_text(json.dumps(ledger,indent=2)+'\n')
