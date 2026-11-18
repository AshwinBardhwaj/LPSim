"""Client watchdog also stops startup failures, which Modal may restart separately."""
from pathlib import Path
import json,queue,re,subprocess,sys,threading,time
root=Path(__file__).resolve().parents[3]
sdk=Path('/tmp/lpsim-modal-sdk/bin/modal');logdir=root/'artifacts/modal-hardware-20261027';logdir.mkdir(exist_ok=True)
cmd=[str(sdk),'run','--timestamps',str(Path(__file__).with_name('modal_hardware.py')),'--output',sys.argv[1]]
if len(sys.argv)>2 and sys.argv[2]=='profile':cmd+=['--profile-only']
logname='profile-followup' if 'profile' in sys.argv else 'corrected-launch'
start=time.monotonic();p=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);q=queue.Queue()
def reader():
    for line in p.stdout:q.put(line)
threading.Thread(target=reader,daemon=True).start();app=None;reason=None
with (logdir/(logname+'.log')).open('w') as log:
    while p.poll() is None or not q.empty():
        try:
            line=q.get(timeout=1);log.write(line);log.flush();print(line,end='',flush=True)
            match=re.search(r'(ap-[A-Za-z0-9]+)',line)
            if match:app=match.group(1)
            if 'Runner failed with exception' in line or 'handle_task_lifecycle_exception' in line:reason='container_initialization_failure'
        except queue.Empty:pass
        if time.monotonic()-start>480:reason='480_second_client_deadline'
        if reason:
            if app:subprocess.run([str(sdk),'app','stop','--yes',app],timeout=45,check=False)
            p.terminate()
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            break
result={'argv':cmd,'app_id':app,'wall_seconds':time.monotonic()-start,'exit_code':p.returncode,'watchdog_stop_reason':reason}
(logdir/(logname+'.json')).write_text(json.dumps(result,indent=2)+'\n')
sys.exit(p.returncode if p.returncode is not None and p.returncode>=0 else 1)
