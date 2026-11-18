"""Stop startup restart loops and enforce an overall client deadline."""
from pathlib import Path
import subprocess,sys,time,queue,threading,re,json
out=Path(sys.argv[1]);phase=sys.argv[2];out.parent.mkdir(parents=True,exist_ok=True)
# Reserve the next full timeout at the more expensive 80GB rate. Completed
# sessions are conservatively charged their function body plus 150 s startup/return.
rate=3*.000694+4*.0000131+8*.00000222
ledger=out.parent
spent_bound=0.0
for statusfile in ledger.glob('*/status.json'):
 data=json.loads(statusfile.read_text());spent_bound+=(data.get('elapsed_s',1100)+150)*rate
reservation=(1100+150)*rate
if spent_bound+reservation>10:raise SystemExit(f'Budget guard: upper accounting {spent_bound:.2f} + reservation {reservation:.2f} exceeds $10')
print(f'Budget upper accounting ${spent_bound:.2f}; reserving ${reservation:.2f}',flush=True)
cmd=['/tmp/lpsim-modal-sdk/bin/modal','run','--timestamps',str(Path(__file__).with_name('modal_study.py')),'--output',str(out),'--phase',phase]
p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1);q=queue.Queue();start=time.monotonic();app=None;reason=None
threading.Thread(target=lambda:[q.put(s) for s in p.stdout],daemon=True).start()
with out.with_suffix('.log').open('w') as f:
 while p.poll() is None or not q.empty():
  try:
   s=q.get(timeout=1);f.write(s);f.flush();print(s,end='',flush=True)
   m=re.search(r'(ap-[A-Za-z0-9]+)',s)
   if m:app=m.group(1)
   if 'Runner failed with exception' in s or 'handle_task_lifecycle_exception' in s:reason='startup_failure'
  except queue.Empty:pass
  if time.monotonic()-start>1350:reason='client_deadline'
  if reason:
   if app:subprocess.run(['/tmp/lpsim-modal-sdk/bin/modal','app','stop','--yes',app],timeout=30)
   p.terminate();break
out.with_suffix('.launch.json').write_text(json.dumps(dict(app_id=app,exit_code=p.poll(),stop_reason=reason,elapsed_s=time.monotonic()-start),indent=2))
sys.exit(p.wait())
