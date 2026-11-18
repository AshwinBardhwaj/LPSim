"""Single authorized cloud launch with cumulative ledger and hard reservation."""
from pathlib import Path
import json,queue,re,subprocess,sys,threading,time
root=Path(__file__).resolve().parents[3];ledger=root/'artifacts/prior-study-20261103'
spent=7.90028261
reservation=6.40
if spent+reservation>15:raise SystemExit('Budget guard: insufficient remaining authorization')
(ledger/'launch-reservation.json').open('x').write(json.dumps(dict(prior_posted=spent,
    reserved=reservation,upper_total=spent+reservation,limit=15),indent=2))
cmd=['/tmp/lpsim-modal-sdk/bin/modal','run','--timestamps',str(Path(__file__).with_name('modal_study.py')),'--output',str(ledger/'cloud')]
p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
q=queue.Queue();start=time.monotonic();app=None;reason=None
threading.Thread(target=lambda:[q.put(s) for s in p.stdout],daemon=True).start()
with (ledger/'cloud.log').open('w') as f:
    while p.poll() is None or not q.empty():
        try:
            s=q.get(timeout=1);f.write(s);f.flush();print(s,end='',flush=True)
            m=re.search(r'(ap-[A-Za-z0-9]+)',s)
            if m:app=m.group(1)
            if 'Runner failed with exception' in s or 'handle_task_lifecycle_exception' in s:reason='startup_failure'
        except queue.Empty:pass
        if time.monotonic()-start>2700:reason='client_deadline'
        if reason:
            if app:subprocess.run(['/tmp/lpsim-modal-sdk/bin/modal','app','stop','--yes',app],timeout=30)
            p.terminate();break
code=p.wait()
(ledger/'launch.json').write_text(json.dumps(dict(app_id=app,exit_code=code,stop_reason=reason,seconds=time.monotonic()-start),indent=2))
sys.exit(code)
