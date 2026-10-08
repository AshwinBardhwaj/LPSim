#!/usr/bin/env python3
"""Render the corrected San Pablo report and offline, frame-based playback."""
import csv, json, math, os, hashlib, shutil, bisect
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/lpsim-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
from shapely import wkt
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/occupancy_fix_20261004'
OUT=ROOT/'reports/san-pablo-occupancy-fix-20261006'
NET=ROOT/'data/networks/san_pablo_university_small'

def read(path):
    with path.open() as f: return list(csv.DictReader(f))
def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True); (OUT/'web').mkdir(exist_ok=True)
    new=json.loads((ART/'summary.json').read_text()); old={s['case']:s for s in json.loads((ROOT/'artifacts/san_pablo_university_20261004/summary.json').read_text())}
    checks=json.loads((ART/'validation.json').read_text()); assert len(checks)==9 and all(c['passed'] for c in checks)
    nodes={int(r['osmid']):(float(r['x']),float(r['y'])) for r in read(NET/'nodes.csv')}
    graph=nx.read_graphml(NET/'source.graphml',node_type=int,force_multigraph=True)
    x0,y0=-122.29216,37.8691
    def xy(p):return [(p[0]-x0)*111320*math.cos(math.radians(y0)),(p[1]-y0)*111320]
    roads=[]
    for r in read(NET/'edges.csv'):
        u,v=int(r['osmid_u']),int(r['osmid_v']);d=next(iter(graph[u][v].values()))
        points=list(wkt.loads(d['geometry']).coords) if 'geometry' in d else [nodes[u],nodes[v]]
        if math.dist(points[0],nodes[u])>math.dist(points[-1],nodes[u]): points.reverse()
        path=[xy(p) for p in points]; offsets=[0]
        for a,b in zip(path,path[1:]): offsets.append(offsets[-1]+math.dist(a,b))
        roads.append(dict(id=int(r['uniqueid']),path=path,offsets=offsets,length=float(r['length']),lanes=int(r['lanes']),name=str(d.get('name','Connector'))))
    (OUT/'web/network.json').write_text(json.dumps(roads,separators=(',',':')))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    pages=[]
    def save(fig,name,caption):
        fig.text(.07,.025,caption,fontsize=8,color='#536779',va='bottom')
        fig.savefig(OUT/'figures'/f'{name}.png',dpi=170);fig.savefig(OUT/'figures'/f'{name}.svg')
        pdf.savefig(fig);plt.close(fig);pages.append((name,caption))
    with PdfPages(OUT/'San-Pablo-Occupancy-Fix.pdf',metadata={'Title':'San Pablo–University | GPU occupancy repair','Author':'LPSim','Subject':'Recorded validation and congestion results; October 6, 2026'}) as pdf:
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,'San Pablo–University',fontsize=30,weight='bold');fig.text(.07,.85,'GPU occupancy repair · results and congestion playback',fontsize=18,color='#087e8b')
        lines=['Prepared October 6, 2026 • corrected recordings from October 4','41 OSM nodes · 91 directed links · NVIDIA Orin · one GPU','600 simulated seconds · 0.5 s timestep · seed 42','20 s red / 20 s green, synchronized at every junction','', '9 / 9 regression scenarios pass the recorded checks','2,676,006 active-vehicle samples checked','Zero duplicate occupied cells after the repair','Zero invalid states and observed red crossings after the repair','', 'What changed','Atomic byte reservations retry on neighboring-byte contention.','Failed moves restore state and occupancy; old cells stay protected for the step.','Car-following equations and braking parameters were not tuned.']
        for i,line in enumerate(lines):fig.text(.075,.76-i*.041,line,fontsize=12 if i not in (5,10) else 15,weight='bold' if i in (5,10) else 'normal')
        save(fig,'01-overview','Scope: sampled one-meter cell ownership, not full vehicle-length collision physics or multi-GPU validation.')
        fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.12,top=.9)
        for r in roads:
            x,y=zip(*r['path']);ax.plot(x,y,color='#cf553c' if r['name'] in ('San Pablo Avenue','University Avenue') else '#a1abb7',lw=2)
        ax.text(15,180,'San Pablo Ave',rotation=100);ax.text(-210,-75,'University Ave',rotation=12)
        ax.set(aspect='equal',xlabel='Meters east of intersection',ylabel='Meters north of intersection',title='Reduced OSM neighborhood: only immediate neighboring streets')
        save(fig,'02-network','© OpenStreetMap contributors. Cached September 17, 2026. Every graph junction has a synthetic light.')
        fig,axs=plt.subplots(3,1,figsize=(11.7,8.3),sharex=True);fig.subplots_adjust(top=.9,bottom=.12,hspace=.35)
        for ax,n in zip(axs,(20,400,1600)):
            data=read(ART/f'network_{n}'/'timeseries.csv');t=[(float(r['timestamp'])-18000)/60 for r in data]
            ax.plot(t,[int(r['active']) for r in data],label='Active',color='#087e8b');ax.plot(t,[int(r['stopped']) for r in data],label='Stopped (<0.5 m/s)',color='#cf553c');ax.set_ylabel('Vehicles');ax.set_title(f'{n:,} requested trips',loc='left');ax.legend(loc='upper right');ax.grid(alpha=.2)
        axs[-1].set_xlabel('Simulation minutes');fig.suptitle('Queue buildup and persistence with cycling lights',fontsize=19)
        save(fig,'03-congestion','Departures occur in the first 3 minutes. Active excludes vehicles waiting to enter. Curves use recorded states.')
        fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.17,top=.88);ax.axis('off')
        table=ax.table(cellText=[[s['case'],str(old[s['case']]['occupied_cell_duplicates']),str(s['occupied_cell_duplicates']),f"{s['completed']} / {s['trips']}"] for s in new],colLabels=['Scenario','Duplicates before','Duplicates after','Completed / requested'],loc='center',cellLoc='center',colWidths=[.35,.2,.2,.25]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.3)
        ax.set_title('Correctness checks: before and after',fontsize=20)
        save(fig,'04-validation','Duplicate counts are repeated vehicle-sample observations, not distinct crashes. Five negative-speed samples also disappeared.')
        fig,axs=plt.subplots(1,2,figsize=(11.7,8.3));fig.subplots_adjust(top=.84,bottom=.2,wspace=.3)
        names=['network_400','network_1600','network_1600_green'];lookup={s['case']:s for s in new};x=list(range(3))
        for dx,source,label,color in [(-.18,old,'Before','#a1abb7'),(.18,lookup,'After','#087e8b')]:axs[0].bar([i+dx for i in x],[source[n]['completed'] for n in names],.36,label=label,color=color)
        axs[0].set_xticks(x,['400 cycle','1,600 cycle','1,600 green']);axs[0].set_title('Completed trips in ten minutes');axs[0].legend()
        cycle=[lookup[f'network_{n}'] for n in (20,100,400,1600)]
        for s in cycle:
            ts=[json.loads((ART/s['case']/f'timing_{i}.json').read_text())['calculation_seconds'] for i in range(3)]
            axs[1].scatter([s['trips']]*3,ts,color='#9ccbd0',s=30)
        axs[1].plot([s['trips'] for s in cycle],[s['calculation_seconds_no_recording_median'] for s in cycle],'o-',color='#087e8b',label='Median of 3');axs[1].set_xscale('log');axs[1].set(xlabel='Requested trips',ylabel='Calculation seconds',title='Corrected runs, recording disabled');axs[1].legend()
        fig.suptitle('Throughput changes with conservative occupancy protection',fontsize=19)
        save(fig,'05-performance','Old-cell protection adds up to one timestep before reuse. Before-fix throughput included overlap. Timings are not pure kernel time.')
        fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.91,'Interpretation and reproducibility',fontsize=25,weight='bold')
        paragraphs=[('What passed','Cell uniqueness, finite/valid states, actual signal phases, red compliance,\nfixed duration, trip accounting and corridor hold/release.'),('What remains outside this validation','Full vehicle-length gaps, between-sample pass-through, realistic turning\nconflicts and multi-GPU boundary behavior are not established.'),('Why throughput decreased','The fix prevents false reservations and protects rollback space. A follower\ncan use a vacated cell on the next timestep. Effects were not isolated individually.'),('Playback','The web viewer replays authoritative recorded states at 0.5 s intervals.\nIt does not run a new simulation or invent continuous vehicle trajectories.'),('Inputs and reruns','artifacts/occupancy_fix_20261004 contains configs, demand, CSV recordings,\nvalidation, source snapshot and binary. tools/build_occupancy_presentation.py\nrebuilds this report and the offline web playback.'),('Performance boundary','GPU counter access was solved using sudo on the earlier build. Those\ncounter samples are not measurements of the corrected binary and are omitted.')]
        for i,(title,body) in enumerate(paragraphs):y=.8-i*.115;fig.text(.07,y,title,fontsize=13,weight='bold');fig.text(.07,y-.035,body,fontsize=11,va='top',linespacing=1.4)
        save(fig,'06-methods','Position validity allows 0.1 mm for CSV/float32 endpoint rounding. Car positions in playback use schematic 3.2 m lanes.')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>San Pablo occupancy fix</title><style>body{font:17px system-ui;max-width:1100px;margin:40px auto;background:#f5f8fa;color:#18333b}img{width:100%}a{color:#087e8b}section{margin:40px 0}</style><h1>San Pablo–University occupancy fix</h1><p><a href="San-Pablo-Occupancy-Fix.pdf">Download PDF</a> · <a href="web/">Open traffic playback</a></p>'+''.join(f'<section><img src="figures/{n}.png" alt="{n}"><p>{c}</p></section>' for n,c in pages))
    # Preserve every recorded active sample in 20-second windows; no interpolation.
    road_by_id={r['id']:r for r in roads};meta={}
    for case in ('network_20','network_400','network_1600','network_1600_green'):
        dest=OUT/'web'/case;dest.mkdir(exist_ok=True); chunks={};count=0
        for r in csv.DictReader((ART/case/'trajectories.csv').open()):
            t=float(r['timestamp'])-18000;frame=round(t*2);bucket=int(t//20);edge=road_by_id[int(r['edge_id'])];lane=int(r['lane_idx']);p=float(r['pos_m'])/edge['length']*edge['offsets'][-1]
            j=max(0,min(len(edge['path'])-2,bisect.bisect_right(edge['offsets'],p)-1));a,b=edge['path'][j:j+2];dist=edge['offsets'][j+1]-edge['offsets'][j];f=(p-edge['offsets'][j])/dist if dist else 0
            lateral=(lane+.5-edge['lanes']/2)*3.2
            x=a[0]+f*(b[0]-a[0])+(b[1]-a[1])/dist*lateral if dist else a[0];y=a[1]+f*(b[1]-a[1])-(b[0]-a[0])/dist*lateral if dist else a[1]
            chunks.setdefault(bucket,{}).setdefault(frame,[]).append([int(r['vehicle_id']),round(x,3),round(y,3),round(float(r['speed']),4),edge['id'],lane]);count+=1
        for bucket,frames in chunks.items():(dest/f'{bucket}.json').write_text(json.dumps(frames,separators=(',',':')))
        phases={}
        for r in csv.DictReader((ART/case/'signals.csv').open()):
            frame=round((float(r['step_start'])-18000)*2)+1;value=int(r['green'])
            if frame in phases:assert phases[frame]==value
            phases[frame]=value
        stats={round((float(r['timestamp'])-18000)*2):[int(r['active']),int(r['stopped']),int(r['max_stopped_per_lane'])] for r in read(ART/case/'timeseries.csv')}
        assert sum(len(v) for frames in chunks.values() for v in frames.values())==lookup[case]['active_vehicle_timesteps']
        meta[case]={'chunks':sorted(chunks),'frames':1200,'signals':phases,'stats':stats,'summary':lookup[case],'samples':count}
        print(case,count,'samples',flush=True)
    (OUT/'web/manifest.json').write_text(json.dumps(meta,separators=(',',':')))
    shutil.copy2(ROOT/'viz/occupancy_viewer.html',OUT/'web/index.html')
    shutil.copy2(ART/'summary.json',OUT/'summary.json');shutil.copy2(ART/'validation.json',OUT/'validation.json')
    print(OUT/'San-Pablo-Occupancy-Fix.pdf')
if __name__=='__main__':main()
