#!/usr/bin/env python3
"""Serve the preserved milestone catalog and existing playback URLs on Tailscale."""
import argparse,socket,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--bind',default='127.0.0.1');args=p.parse_args()
 archive=ROOT.parent/'research-archive';runtime=archive/'runtime';runtime.mkdir(exist_ok=True)
 targets={8090:ROOT/'reports/san-pablo-occupancy-fix-20261006',8091:ROOT/'reports/san-pablo-junctions-20261006',8092:ROOT/'reports/san-pablo-platoons-20261007',8093:archive/'site'}
 for port,directory in targets.items():
  if not directory.exists():raise SystemExit(f'Missing directory: {directory}')
  with socket.socket() as probe:
   if probe.connect_ex((args.bind,port))==0:
    print(f'Port {port} already has a listener; leaving it unchanged');continue
  with (runtime/f'server-{port}.log').open('ab') as log:
   child=subprocess.Popen(['python3','-m','http.server',str(port),'--bind',args.bind,'--directory',str(directory)],stdout=log,stderr=log,start_new_session=True)
  (runtime/f'server-{port}.pid').write_text(str(child.pid))
  print(f'http://{args.bind}:{port}/ PID {child.pid}')
if __name__=='__main__':main()
