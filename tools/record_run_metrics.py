#!/usr/bin/env python3
"""Record a command's wall runtime and Jetson telemetry for the offline report."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time


def parse_tegrastats(line):
    patterns={'gpu_util_pct':r'GR3D_FREQ\s+(\d+)%','ram_used_mb':r'RAM\s+(\d+)/',
              'temperature_c':r'\bGPU@([\d.]+)C','power_w':r'\b(?:VDD_IN|POM_5V_IN)\s+(\d+)(?:mW)?/'}
    values={}
    for key,pattern in patterns.items():
        match=re.search(pattern,line)
        if match: values[key]=float(match[1])/(1000 if key=='power_w' else 1)
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cwd',type=Path,default=Path.cwd())
    p.add_argument('--report-run-dir',type=Path,help='Generate a report after success from this run package (inputs/network, routes.csv, trajectories.csv)')
    p.add_argument('--no-telemetry',action='store_true')
    p.add_argument('command',nargs=argparse.REMAINDER)
    args=p.parse_args(); command=args.command
    if command and command[0]=='--': command=command[1:]
    if not command: p.error('Supply -- COMMAND [arguments]')
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); telemetry=None; thread=None
    manifest={'command':command,'cwd':str(args.cwd.resolve()),'started_utc':datetime.now(timezone.utc).isoformat(),
        'telemetry':'disabled' if args.no_telemetry else 'unavailable',
        'power_definition':'Total module input power (VDD_IN/POM_5V_IN), not GPU-only power',
        'memory_definition':'Shared system RAM, not dedicated GPU memory',
        'gpu_definition':'GR3D activation percentage, not warp occupancy',
        'capture_environment':{k:os.environ[k] for k in ('LPSIM_TRAJECTORIES','LPSIM_STEP_METRICS') if k in os.environ}}
    def collect():
        fields=['wall_seconds','gpu_util_pct','ram_used_mb','temperature_c','power_w']
        with (out/'telemetry.csv').open('w',newline='') as f, (out/'tegrastats.jsonl').open('w') as raw:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for line in telemetry.stdout:
                elapsed=time.monotonic()-started
                raw.write(json.dumps({'wall_seconds':elapsed,'text':line.rstrip()})+'\n')
                values=parse_tegrastats(line)
                if values: writer.writerow({'wall_seconds':elapsed,**values}); f.flush()
    error_log=(out/'telemetry-errors.log').open('w')
    if not args.no_telemetry and shutil.which('tegrastats'):
        telemetry=subprocess.Popen(['tegrastats','--interval','1000'],stdout=subprocess.PIPE,stderr=error_log,text=True)
        thread=threading.Thread(target=collect); thread.start(); manifest['telemetry']='started; check rows and errors'
    try:
        with (out/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=args.cwd,stdout=log,stderr=subprocess.STDOUT)
        manifest['exit_code']=result.returncode
    finally:
        manifest['wall_seconds']=time.monotonic()-started
        if telemetry:
            telemetry.terminate()
            try: telemetry.wait(timeout=5)
            except subprocess.TimeoutExpired: telemetry.kill(); telemetry.wait()
            thread.join(timeout=5)
        error_log.close()
        manifest['files']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()}
        (out/'capture.json').write_text(json.dumps(manifest,indent=2))
    print(f'Captured run in {out}; exit={result.returncode}')
    if result.returncode == 0 and args.report_run_dir:
        generator=Path(__file__).resolve().with_name('generate_run_report.py')
        report=subprocess.run([sys.executable,str(generator),
            '--run-dir',str(args.report_run_dir.resolve()),'--capture',str(out),
            '--output',str(out/'report')])
        if report.returncode:
            print('Simulation succeeded but report generation failed; captured data is preserved.')
            raise SystemExit(report.returncode)
    raise SystemExit(result.returncode)


if __name__=='__main__': main()
