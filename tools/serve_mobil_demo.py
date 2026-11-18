#!/usr/bin/env python3
"""Run an isolated Berkeley MOBIL scenario and serve live frames on loopback."""
import argparse, bisect, csv, json, math, os, random, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--port', type=int, default=8088)
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    if (out/'scenario.txt').exists():
        raise SystemExit('Use a fresh output directory to preserve existing runs')
    net = ROOT/'data/networks/berkeley_junction_study'
    geometry = json.loads((net/'geometry.json').read_text())
    roads, moves = geometry['roads'], geometry['movements']
    routes = json.loads((net/'route_pool.json').read_text())
    rng = random.Random(1811)
    with (out/'scenario.txt').open('w') as f:
        f.write(f'{net / "model.txt"} 8000 600 0.25 0\n')
        for i in range(8000):
            r = rng.choice(routes)
            f.write(' '.join(map(str, [round(rng.uniform(0,120),3), len(r['edges']), *r['edges'], *r['lanes'], *r['moves'], -1]))+'\n')
    paths = {}
    for m in moves:
        offsets = [0.]
        for a,b in zip(m['path'],m['path'][1:]): offsets.append(offsets[-1]+math.dist(a,b))
        paths[m['id']] = offsets
    def point(row):
        t, ident, state, edge, lane, mid, pos, speed = row
        ident,state,edge,lane,mid = map(int,(ident,state,edge,lane,mid)); pos=float(pos)
        if state == 2:
            m=moves[mid]; path=m['path']; offsets=paths[mid]
            d=max(0,min(pos/m['length']*offsets[-1],offsets[-1])); j=min(len(path)-2,max(0,bisect.bisect_right(offsets,d)-1))
            frac=(d-offsets[j])/max(1e-9,offsets[j+1]-offsets[j])
            x,y=[path[j][k]+frac*(path[j+1][k]-path[j][k]) for k in (0,1)]
        else:
            r=roads[edge]; f=max(0,min(1,pos/r['length'])); dx,dy=r['tangent']; off=(lane+.5-r['lanes']/2)*3.2
            x=r['start'][0]+f*(r['end'][0]-r['start'][0])+dy*off
            y=r['start'][1]+f*(r['end'][1]-r['start'][1])-dx*off
        return [ident,round(x,2),round(y,2),round(float(speed),2),lane]
    frames={}; lock=threading.Lock(); process=None; reader_error=None
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            query=urlparse(self.path)
            if query.path=='/':
                data=(ROOT/'viz/mobil_live.html').read_bytes(); content='text/html'
            elif query.path=='/network':
                data=json.dumps([{'a':r['start'],'b':r['end'],'lanes':r['lanes']} for r in roads]).encode(); content='application/json'
            elif query.path=='/frame':
                requested=float(parse_qs(query.query).get('time',['-1'])[0])
                with lock:
                    times=sorted(frames); selected=times[-1] if times else 0
                    if requested>=0 and times: selected=times[max(0,bisect.bisect_right(times,requested)-1)]
                    cars=frames.get(selected,[])
                metrics=None
                if process and process.poll() is not None and (out/'metrics.json').exists():
                    metrics=json.loads((out/'metrics.json').read_text())
                data=json.dumps(dict(time=selected,latest=times[-1] if times else 0,cars=cars,
                    running=process is not None and process.poll() is None,exit_code=process.poll() if process else None,
                    metrics=metrics,error=reader_error,trips=8000,mobil=True)).encode(); content='application/json'
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header('Content-Type',content); self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(data)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    env=os.environ.copy(); env.update(LPSIM_MOBIL='1',LPSIM_COMPACT_MOTION='1',LPSIM_LEADER_INDEX='1',LPSIM_MOTION_CONTEXT='3',LPSIM_REQUEST_INDEX='1',LPSIM_TRAJECTORY_STRIDE='2')
    for key in ('LPSIM_RESTART','LPSIM_JUNCTION_PARALLEL','LPSIM_JUNCTION_EXCLUSIVE','LPSIM_MOBIL_POLITENESS','LPSIM_MOBIL_SAFE_BRAKING','LPSIM_MOBIL_THRESHOLD','LPSIM_MOBIL_JUNCTION_MARGIN'):
        env.pop(key,None)
    (out/'launch.json').write_text(json.dumps({'binary':str(args.binary.resolve()),'vehicles':8000,'duration':600,'env':{k:v for k,v in env.items() if k.startswith('LPSIM_')},'url':f'http://localhost:{args.port}'},indent=2))
    log=(out/'simulation.log').open('w')
    process=subprocess.Popen([str(args.binary.resolve()),str(out/'scenario.txt')],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    def read_frames():
        nonlocal reader_error
        try:
            path=out/'states.csv'
            while not path.exists():
                if process.poll() is not None: return
                time.sleep(.1)
            pending=''; current=None; batch=[]
            with path.open() as f:
                while True:
                    line=f.readline()
                    if not line:
                        if process.poll() is not None: break
                        time.sleep(.1); continue
                    pending+=line
                    if not pending.endswith('\n'): continue
                    row=next(csv.reader([pending])); pending=''
                    if row[0]=='time': continue
                    stamp=float(row[0])
                    if current is not None and stamp!=current:
                        with lock: frames[current]=batch
                        batch=[]
                    current=stamp; batch.append(point(row))
                if current is not None:
                    with lock: frames[current]=batch
        except Exception as exc:
            reader_error=str(exc)
    threading.Thread(target=read_frames,daemon=True).start()
    print(f'Serving http://localhost:{args.port}; simulation PID {process.pid}',flush=True)
    try: server.serve_forever()
    finally:
        if process.poll() is None: process.terminate(); process.wait()
        server.server_close(); log.close()
if __name__=='__main__': main()
