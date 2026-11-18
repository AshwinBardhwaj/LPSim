"""Optional separate hardware-counter samples, never benchmark timings."""
from pathlib import Path
import json,os,shutil,subprocess,sys,time,hashlib
out=Path(sys.argv[1]);ncu=shutil.which('ncu');records=[]
if not ncu:
    options=list(Path('/usr/local').glob('**/ncu'))
    ncu=str(options[0]) if options else None
if not ncu:
    (out/'profiles.json').write_text(json.dumps(dict(available=False,reason='ncu absent')));sys.exit(0)
chosen=json.loads((out/'backend-selection.json').read_text())['chosen']
metrics=['gpu__time_duration.sum','sm__throughput.avg.pct_of_peak_sustained_elapsed',
         'sm__warps_active.avg.pct_of_peak_sustained_active',
         'smsp__warps_eligible.avg.per_cycle_active',
         'smsp__thread_inst_executed_per_inst_executed.ratio',
         'smsp__issue_active.avg.pct_of_peak_sustained_active',
         'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct',
         'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
         'lts__throughput.avg.pct_of_peak_sustained_elapsed','launch__registers_per_thread']
for mode in sorted({0,chosen}):
    d=out/f'profile-mode{mode}';d.mkdir()
    cmd=[ncu,'--clock-control','none','--target-processes','all','--kernel-name',
         'regex:packLocal|packSlots|gatherLocal|gatherSlots|advanceCompact|arbitrateIndexedJunctions',
         '--launch-count','18','--metrics',','.join(metrics),'--csv','--page','raw',
         '--log-file',str(d/'metrics.csv'),'--export',str(d/'profile'),
         '/workspace/bin/local','/workspace/inputs/models/sf_partial_local_scaling/model.txt',
         '/workspace/inputs/sf_98304/snapshot.bin','/workspace/inputs/sf_98304/paper_balanced_3.txt',
         '1',str(d),'3','0','0']
    start=time.monotonic()
    with (d/'log.txt').open('w') as log:
        try:code=subprocess.run(cmd,env=dict(os.environ,LPSIM_EXCHANGE_MODE=str(mode)),stdout=log,stderr=subprocess.STDOUT,timeout=45).returncode
        except subprocess.TimeoutExpired:code=-1
    row=dict(mode=mode,exit_code=code,seconds=time.monotonic()-start,argv=cmd)
    if (d/'final.bin').exists():
        row['final_sha256']=hashlib.sha256((d/'final.bin').read_bytes()).hexdigest();(d/'final.bin').unlink()
    records.append(row);(out/'profiles.json').write_text(json.dumps(records,indent=2))
