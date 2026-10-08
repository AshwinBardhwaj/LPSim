#!/usr/bin/env python3
"""Generate an offline HTML/PDF traffic and compute report from recorded outputs."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent if (Path(__file__).resolve().parent/'process_routes.py').exists() else ROOT/'viz'))
from process_routes import load_edges, load_routes


def split_interval(route, prefix, edges, previous, current):
    """Allocate observed dt across traversed roads by route distance (an estimate)."""
    t0, i0, _, p0, _ = previous
    t1, i1, _, p1, _ = current
    dt = t1 - t0
    s0, s1 = prefix[i0] + p0, prefix[i1] + p1
    ds = s1 - s0
    if dt <= 0 or i1 < i0 or ds < -0.01 or ds / dt > 45:
        return []
    if any(edges[route[j]].v != edges[route[j+1]].u for j in range(i0, i1)):
        return []
    if ds <= 0:
        return [(route[i0], t0, t1, 0.0)]
    parts, clock = [], t0
    for j in range(i0, i1 + 1):
        length = max(0, min(s1, prefix[j+1]) - max(s0, prefix[j]))
        if length:
            end = clock + dt * length / ds
            parts.append((route[j], clock, end, length))
            clock = end
    return parts


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def aggregate(network, routes_file, trajectories, output, bin_seconds=300):
    edges, routes = load_edges(network), load_routes(routes_file)
    with (network/'edges.csv').open() as f:
        free = {int(r['uniqueid']): float(r['speed_mph']) * 0.44704 for r in csv.DictReader(f)}
    segment_bins=defaultdict(lambda: Counter())
    roads, bins, vehicles, quality = defaultdict(lambda: Counter()), defaultdict(lambda: Counter()), [], Counter()
    with tempfile.TemporaryDirectory() as tmp:
        con = sqlite3.connect(str(Path(tmp)/'sort.sqlite'))
        con.execute('CREATE TABLE samples(vid TEXT,t REAL,idx INTEGER,eid INTEGER,pos REAL,speed REAL)')
        with trajectories.open() as f:
            reader = csv.DictReader(f)
            required = {'vehicle_id','timestamp','route_index','edge_id','pos_m','speed','edge_id_kind'}
            if not required <= set(reader.fieldnames or []): raise ValueError('Authoritative trajectory schema required')
            batch = []
            for r in reader:
                if r['edge_id_kind'] != 'uniqueid': raise ValueError('Physical edge IDs required')
                values = (r['vehicle_id'],float(r['timestamp']),int(r['route_index']),int(r['edge_id']),float(r['pos_m']),float(r['speed']))
                if not all(math.isfinite(values[k]) for k in (1,4,5)): raise ValueError('Non-finite input')
                batch.append(values)
                if len(batch) == 50000:
                    con.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?)',batch); batch.clear()
            con.executemany('INSERT INTO samples VALUES (?,?,?,?,?,?)',batch)
        con.commit()
        print('Sorting samples on disk…',flush=True)
        con.execute('CREATE INDEX chronological ON samples(vid,t)')
        for vid, group in itertools.groupby(con.execute('SELECT * FROM samples ORDER BY vid,t'),key=lambda r:r[0]):
            route = routes.get(vid)
            if not route or any(e not in edges for e in route):
                quality['vehicles_missing_routes'] += 1
                continue
            prefix=[0.0]
            for e in route: prefix.append(prefix[-1]+edges[e].length)
            total=Counter(); entry_times={}; previous=None; first=None; last=None; speed_sum=0; count=0
            for t, equal_time in itertools.groupby(group,key=lambda r:r[1]):
                observed=list(equal_time); unique=set(r[1:] for r in observed)
                quality['input_samples'] += len(observed)
                quality['duplicates'] += len(observed)-len(unique)
                if len(unique)!=1:
                    previous=None; entry_times.clear(); quality['conflicting_timestamps']+=1; continue
                row=unique.pop(); _, idx, eid, pos, speed=row
                if not 0<=idx<len(route) or eid!=route[idx] or not -0.01<=pos<=edges[eid].length+0.01:
                    previous=None; entry_times.clear(); quality['invalid_samples']+=1; continue
                row=(t,idx,eid,min(edges[eid].length,max(0,pos)),speed)
                first=t if first is None else first; last=t; speed_sum+=speed; count+=1
                quality['accepted_samples']+=1
                if previous:
                    parts=split_interval(route,prefix,edges,previous,row)
                    if not parts:
                        quality['invalid_intervals']+=1
                        entry_times.clear()
                    elif idx > previous[1]:
                        s0=prefix[previous[1]]+previous[3]
                        ds=prefix[idx]+row[3]-s0
                        for boundary in range(previous[1]+1,idx+1):
                            crossing=previous[0]+(t-previous[0])*(prefix[boundary]-s0)/ds
                            if boundary-1 in entry_times:
                                road=route[boundary-1]
                                roads[road]['estimated_traversal_s']+=crossing-entry_times[boundary-1]
                                roads[road]['estimated_traversals']+=1
                            entry_times={boundary:crossing}
                    for road,a,b,ds in parts:
                        dt=b-a; stopped=dt if ds/dt<0.1 else 0
                        delay=max(0,dt-ds/free[road]) if free[road]>0 else 0
                        for target in (total,roads[road]):
                            target['distance_m']+=ds; target['time_s']+=dt
                            target['stopped_s']+=stopped; target['delay_s']+=delay
                        cursor=a
                        while cursor < b-1e-9:
                            bucket=math.floor(cursor/bin_seconds)
                            end=min(b,(bucket+1)*bin_seconds); fraction=(end-cursor)/dt
                            for k,value in [('distance_m',ds),('time_s',dt),('stopped_s',stopped),('delay_s',delay)]:
                                bins[bucket][k]+=value*fraction
                                segment_bins[(road,bucket)][k]+=value*fraction
                            cursor=end
                previous=row
            if count:
                vehicles.append({'vehicle_id':vid,'first_sample_s':first,'last_sample_s':last,
                    'observed_span_s':last-first,'valid_observed_time_s':total['time_s'],
                    'observed_vmt':total['distance_m']/1609.344,'observed_vht':total['time_s']/3600,
                    'distance_time_speed_mps':total['distance_m']/total['time_s'] if total['time_s'] else '',
                    'mean_sampled_speed_mps':speed_sum/count,'observed_stopped_s':total['stopped_s'],
                    'estimated_free_flow_delay_s':total['delay_s'],'samples':count})
        con.close()
    write_csv(output/'vehicles.csv',vehicles,list(vehicles[0]) if vehicles else ['vehicle_id'])
    segment_rows=[]
    for eid,c in sorted(roads.items()):
        segment_rows.append({'edge_id':eid,'length_m':edges[eid].length,'lanes':edges[eid].lanes,
            'observed_vmt':c['distance_m']/1609.344,'observed_vht':c['time_s']/3600,
            'distance_time_speed_mps':c['distance_m']/c['time_s'] if c['time_s'] else 0,
            'observed_stopped_s':c['stopped_s'],'estimated_free_flow_delay_s':c['delay_s'],
            'estimated_complete_traversals':int(c['estimated_traversals']),
            'mean_estimated_traversal_s':c['estimated_traversal_s']/c['estimated_traversals'] if c['estimated_traversals'] else '',
            'length_over_mean_speed_s':edges[eid].length*c['time_s']/c['distance_m'] if c['distance_m'] else ''})
    write_csv(output/'segments.csv',segment_rows,list(segment_rows[0]) if segment_rows else ['edge_id'])
    coverage_start=min(v['first_sample_s'] for v in vehicles) if vehicles else 0
    coverage_end=max(v['last_sample_s'] for v in vehicles) if vehicles else 0
    def exposure(bucket):
        return max(0,min(coverage_end,(bucket+1)*bin_seconds)-max(coverage_start,bucket*bin_seconds))
    segment_time_rows=({'edge_id':road,'simulation_time_s':b*bin_seconds,'coverage_seconds':exposure(b),
        'observed_vmt':c['distance_m']/1609.344,'observed_vht':c['time_s']/3600,
        'mean_observed_vehicles':c['time_s']/exposure(b) if exposure(b) else 0,
        'distance_time_speed_mps':c['distance_m']/c['time_s'] if c['time_s'] else 0}
        for (road,b),c in sorted(segment_bins.items()))
    write_csv(output/'segment_time_bins.csv',segment_time_rows,['edge_id','simulation_time_s','coverage_seconds','observed_vmt','observed_vht','mean_observed_vehicles','distance_time_speed_mps'])
    time_rows=[{'simulation_time_s':b*bin_seconds,'coverage_seconds':exposure(b),'observed_vmt':c['distance_m']/1609.344,
        'observed_vht':c['time_s']/3600,'mean_observed_vehicles':c['time_s']/exposure(b) if exposure(b) else 0,
        'distance_time_speed_mps':c['distance_m']/c['time_s'] if c['time_s'] else 0,
        'observed_stopped_vehicle_seconds':c['stopped_s']} for b,c in sorted(bins.items())]
    write_csv(output/'time_bins.csv',time_rows,list(time_rows[0]) if time_rows else ['simulation_time_s'])
    summary={'vehicles_observed':len(vehicles),'segments_observed':len(roads),
        'observed_vmt':sum(v['observed_vmt'] for v in vehicles),
        'observed_vht':sum(v['observed_vht'] for v in vehicles),'quality':dict(quality),
        'bin_seconds':bin_seconds,'stop_threshold_mps':0.1}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    return vehicles,segment_rows,time_rows,edges,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,default=ROOT/'artifacts/berkeley-lane-review')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--from-tables',type=Path,help='Reuse hash-verified tables from a prior report for re-rendering')
    p.add_argument('--capture',type=Path,help='Recorder directory; discover telemetry and optional compute CSVs')
    p.add_argument('--bin-seconds',type=float,default=300)
    p.add_argument('--telemetry',type=Path)
    p.add_argument('--steps',type=Path)
    p.add_argument('--partitions',type=Path)
    p.add_argument('--warps',type=Path,help='Timestamped profiler metrics CSV; do not fabricate timestamps from launch IDs')
    p.add_argument('--ncu',type=Path,help='Nsight Compute raw CSV export')
    args=p.parse_args()
    if not math.isfinite(args.bin_seconds) or args.bin_seconds <= 0:
        p.error('--bin-seconds must be positive and finite')
    if args.capture:
        for name in ('telemetry','steps','partitions','warps'):
            candidate=(args.capture if args.capture.is_dir() else args.capture.parent)/(name+'.csv')
            if getattr(args,name) is None and candidate.is_file(): setattr(args,name,candidate)
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    inputs=args.run_dir/'inputs'
    source=[inputs/'network/nodes.csv',inputs/'network/edges.csv',inputs/'routes.csv',inputs/'trajectories.csv']
    print('Reading recorded run…',flush=True)
    if args.from_tables:
        prior=json.loads((args.from_tables/'manifest.json').read_text())
        if prior['summary']['bin_seconds'] != args.bin_seconds: raise ValueError('Cached bin width differs')
        for path in source:
            if prior['inputs'].get(str(path.resolve())) != digest(path): raise ValueError('Cached input hash mismatch')
        for name in ('vehicles.csv','segments.csv','time_bins.csv','segment_time_bins.csv','summary.json'):
            path=args.from_tables/name
            if prior['outputs'].get(name) != digest(path): raise ValueError('Cached table hash mismatch')
            shutil.copy2(path,out/name)
        def table(name):
            with (out/name).open() as f:
                return [{k: (v if k=='vehicle_id' or v=='' else float(v)) for k,v in r.items()} for r in csv.DictReader(f)]
        vehicles,segments,bins=table('vehicles.csv'),table('segments.csv'),table('time_bins.csv')
        for row in segments: row['edge_id']=int(row['edge_id'])
        edges=load_edges(inputs/'network'); summary=json.loads((out/'summary.json').read_text())
    else:
        vehicles,segments,bins,edges,summary=aggregate(inputs/'network',source[2],source[3],out,args.bin_seconds)
    if not vehicles: raise ValueError('No valid vehicle observations')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator
    import numpy as np
    plt.rcParams.update({'figure.dpi':140,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    figures=[]; availability={}
    if args.capture:
        capture_path=args.capture/'capture.json' if args.capture.is_dir() else args.capture
        capture=json.loads(capture_path.read_text()); source.append(capture_path)
        availability['whole_command_runtime']=f"{capture['wall_seconds']:.3f} wall seconds; exit code {capture.get('exit_code', 'unknown')}; includes setup/routing/I/O"
    notes=('Observed coverage only: insertion and completion events were not recorded. '
        'Trip spans exclude unobserved endpoints; they are not complete trip travel times. '
        'Road-boundary times are allocated proportionally to route distance within each sample interval. '
        'VMT/VHT exclude invalid gaps. Segment length/mean speed is a travel-time proxy, not mean measured traversal time. Separate traversal estimates require both boundary crossings within the recording. '
        'Stopped means interval speed <0.1 m/s. Free-flow delay uses each road speed limit. '
        'No claim of correct signal, merge or lane-change behavior follows from these figures.')
    pdf=PdfPages(out/'report.pdf')
    def save(fig,name,caption):
        fig.tight_layout(); fig.savefig(out/(name+'.png')); pdf.savefig(fig); plt.close(fig)
        figures.append((name,caption))
    fig,ax=plt.subplots(figsize=(10,7)); ax.axis('off')
    ax.text(0,1,'LPSim • Recorded-run performance report',va='top',fontsize=20)
    ax.text(0,.84,f"Vehicles: {summary['vehicles_observed']:,}\nRoad segments: {summary['segments_observed']:,}\nObserved VMT: {summary['observed_vmt']:,.1f} vehicle-miles\nObserved VHT: {summary['observed_vht']:,.1f} vehicle-hours",va='top',fontsize=14)
    import textwrap
    ax.text(0,.48,textwrap.fill(notes,95),va='top',linespacing=1.6)
    save(fig,'overview','Coverage and metric definitions; read before interpreting the figures.')
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,key,title in zip(axes.flat,['observed_span_s','observed_vmt','distance_time_speed_mps','observed_stopped_s'],['Observed vehicle duration (s)','Observed vehicle distance (miles)','Distance/time speed (m/s)','Observed stopped time (s)']):
        ax.hist([v[key] for v in vehicles if v[key]!=''],bins=45,color='#287c8e'); ax.set_title(title); ax.set_ylabel('Vehicles')
    save(fig,'vehicle_distributions','Per-vehicle distributions. Duration is first-to-last recorded sample, not insertion-to-completion travel time.')
    plot_bins=[b for b in bins if b['coverage_seconds'] >= summary['bin_seconds']*.95]
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for ax,key,label in zip(axes,['distance_time_speed_mps','mean_observed_vehicles','observed_vmt','observed_vht'],['Network speed (m/s)','Mean observed vehicles','VMT per bin','VHT per bin']):
        ax.plot([b['simulation_time_s']/3600 for b in plot_bins],[b[key] for b in plot_bins]); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Simulation clock (hours)')
    save(fig,'traffic_over_time',f'{args.bin_seconds:g}-second aggregates; bins with less than 95% temporal coverage omitted from this figure. Speed = accumulated distance / accumulated vehicle time.')
    fig,axes=plt.subplots(2,2,figsize=(12,11))
    for ax,key,label in zip(axes.flat,['distance_time_speed_mps','observed_vht','observed_vmt','length_over_mean_speed_s'],['Road speed (m/s)','Road VHT (vehicle-hours)','Road VMT (vehicle-miles)','Road travel-time proxy (s)']):
        collection=LineCollection([edges[r['edge_id']].points for r in segments],array=np.array([r[key] if r[key]!='' else np.nan for r in segments]),cmap='viridis',linewidths=1)
        ax.add_collection(collection); ax.autoscale(); ax.xaxis.set_major_locator(MaxNLocator(4)); ax.set_aspect(1/math.cos(math.radians(37.87))); ax.set_title(label); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); fig.colorbar(collection,ax=ax,shrink=.6)
    save(fig,'road_maps','Road-level spatial distribution. Endpoint geometry is used where WKT curves are absent.')
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for ax,key,label in zip(axes,['observed_vmt','observed_vht'],['Cumulative VMT (vehicle-miles)','Cumulative VHT (vehicle-hours)']):
        ax.plot([b['simulation_time_s']/3600 for b in bins],np.cumsum([b[key] for b in bins])); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[-1].set_xlabel('Simulation clock (hours; bin start)')
    save(fig,'cumulative_vmt_vht','Accumulated observed distance and time. Includes all recorded bins, including partial boundary bins.')
    fig,ax=plt.subplots(figsize=(11,6)); top=sorted(segments,key=lambda r:r['estimated_free_flow_delay_s'],reverse=True)[:15]
    ax.barh([str(r['edge_id']) for r in top][::-1],[r['estimated_free_flow_delay_s']/3600 for r in top][::-1]); ax.set_xlabel('Estimated excess vehicle-hours above speed-limit baseline'); ax.set_ylabel('Physical edge ID')
    save(fig,'road_delay','Highest accumulated estimated delay, not necessarily the slowest roads; high demand also raises totals.')
    schemas={'telemetry':('wall_seconds',['gpu_util_pct','ram_used_mb','temperature_c','power_w']),
        'steps':('wall_seconds',['step_ms','simulation_seconds']),
        'partitions':('simulation_seconds',['active_vehicles','migrations','transfer_bytes']),
        'warps':('wall_seconds',['achieved_occupancy_pct','active_warps','eligible_warps'])}
    for name,(x,ys) in schemas.items():
        path=getattr(args,name)
        if not path:
            availability[name]='Not recorded / not supplied'; continue
        source.append(path)
        with path.open() as f: rows=list(csv.DictReader(f))
        if not rows:
            availability[name]='No samples recorded'; continue
        if x not in rows[0]: raise ValueError(f'{name} requires {x}')
        columns=[y for y in ys if any(r.get(y,'')!='' for r in rows)]
        if not columns: availability[name]='No usable measurements'; continue
        fig,axes=plt.subplots(len(columns),1,figsize=(11,3*len(columns)),squeeze=False)
        for ax,y in zip(axes.flat,columns):
            groups=defaultdict(list)
            for r in rows:
                if r.get(y,'')!='': groups[r.get('partition_id') or r.get('gpu_id') or 'all'].append(r)
            for group,data in groups.items():
                data.sort(key=lambda r:float(r[x])); ax.plot([float(r[x]) for r in data],[float(r[y]) for r in data],label=group)
            labels={'gpu_util_pct':'GPU active time (%)','ram_used_mb':'Shared RAM used (MB)','temperature_c':'GPU temperature (°C)','power_w':'Module input power (W)','step_ms':'Step duration (ms)','simulation_seconds':'Simulation time (s)','wall_seconds':'Elapsed wall time (s)','active_vehicles':'Active vehicles','migrations':'Vehicle migrations','transfer_bytes':'Transferred bytes','achieved_occupancy_pct':'Achieved occupancy (%)','active_warps':'Active warps','eligible_warps':'Eligible warps'}
            ax.set_ylabel(labels.get(y,y)); ax.set_xlabel(labels.get(x,x)); ax.legend(title='GPU/partition')
        availability[name]=f'{len(rows)} rows supplied'
        save(fig,name,'Measured computational data. Wall-clock and simulation time are distinct. RAM is shared system RAM on Jetson, not dedicated GPU memory.')
    availability['warp_metrics']='Not recorded / no Nsight Compute CSV supplied'
    if args.ncu:
        source.append(args.ncu)
        with args.ncu.open() as f: lines=f.readlines()
        header=next((i for i,line in enumerate(lines) if '"Metric Name"' in line and '"Metric Value"' in line),None)
        if header is None: raise ValueError('Expected Nsight Compute --csv --page raw export')
        metrics=defaultdict(list)
        for r in csv.DictReader(lines[header:]):
            if any(k in r['Metric Name'] for k in ('warps_active','warps_eligible','warp_issue','throughput')):
                try: value=float(r['Metric Value'].replace(',',''))
                except ValueError: continue
                metrics[(r['Metric Name'],r['Metric Unit'])].append((r.get('ID',''),value))
        availability['warp_metrics']=f'{len(metrics)} metric series supplied (profiled launches, not continuous time)'
        for i,((name,unit),values) in enumerate(metrics.items()):
            fig,ax=plt.subplots(figsize=(11,4)); ax.plot(range(len(values)),[v for _,v in values],marker='o'); ax.set_title(name); ax.set_ylabel(unit); ax.set_xlabel('Profiled launch sequence (not simulation time)')
            save(fig,f'profile_{i}','Profiler-derived metrics. Profiling may perturb runtime; these are not unprofiled utilization measurements.')
    fig,ax=plt.subplots(figsize=(10,6)); ax.axis('off'); ax.text(0,1,'Computational measurement coverage',fontsize=18,va='top')
    ax.text(0,.85,'\n\n'.join(f'{k}: {v}' for k,v in availability.items()),va='top',wrap=True)
    ax.text(0,.25,'No missing measurements are replaced with zero.\nSingle-GPU runs cannot demonstrate multi-GPU scaling.\nWarp occupancy is not equivalent to GPU utilization.',va='top')
    save(fig,'compute_coverage','Missing telemetry requires a new instrumented run; it cannot be reconstructed from vehicle trajectories.')
    pdf.close()
    if args.capture and (args.capture/'capture.json').is_file():
        source.append(args.capture/'capture.json')
    table=''.join(f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>' for k,v in availability.items())
    body=''.join(f'<section><img src="{name}.png"><p>{html.escape(caption)}</p></section>' for name,caption in figures)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>LPSim run report</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px;color:#18333b}img{width:100%}section{margin:40px 0}td{padding:10px;border-bottom:1px solid #ddd}</style><h1>LPSim run report</h1><p>'+html.escape(notes)+'</p><p><a href="report.pdf">Download PDF</a> · <a href="vehicles.csv">Vehicles CSV</a> · <a href="segments.csv">Road segments CSV</a> · <a href="time_bins.csv">Time series CSV</a> · <a href="segment_time_bins.csv">Road time series CSV</a></p><table>'+table+'</table>'+body)
    manifest={'inputs':{str(p.resolve()):digest(p) for p in source},'summary':summary,'compute_availability':availability,
        'python':sys.version,'matplotlib':matplotlib.__version__,'command':sys.argv,
        'limitations':notes,'source_provenance':'Existing input hashes; historical simulator build/seed not inferred'}
    shutil.copy2(__file__,out/'generator.py')
    import process_routes
    shutil.copy2(process_routes.__file__,out/'process_routes.py')
    manifest['outputs']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Report ready: {out / "report.html"}',flush=True)


if __name__=='__main__': main()
