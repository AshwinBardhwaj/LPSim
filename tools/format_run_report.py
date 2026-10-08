#!/usr/bin/env python3
"""Create a paginated PDF and self-contained browser reader from verified run tables."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator
import numpy as np

from generate_run_report import digest, load_edges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tables', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    src, out = args.tables.resolve(), args.output.resolve()
    manifest = json.loads((src/'manifest.json').read_text())
    for filename in ('vehicles.csv', 'segments.csv', 'time_bins.csv', 'segment_time_bins.csv', 'summary.json'):
        if digest(src/filename) != manifest['outputs'][filename]:
            raise ValueError(f'Table checksum mismatch: {filename}')
    for filename, expected in manifest['inputs'].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f'Input checksum mismatch: {filename}')
    print('Verified full-run inputs and aggregate tables.', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    def read(name):
        with (src/name).open() as f:
            return [{k: (v if k == 'vehicle_id' else float(v) if v else np.nan)
                     for k, v in r.items()} for r in csv.DictReader(f)]
    vehicles, roads, bins = read('vehicles.csv'), read('segments.csv'), read('time_bins.csv')
    summary = json.loads((src/'summary.json').read_text())
    network = next(Path(p).parent for p in manifest['inputs'] if p.endswith('/network/edges.csv'))
    edges = load_edges(network)
    color, ink, muted = '#087e8b', '#17324d', '#536779'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': ink, 'text.color': ink, 'axes.titlecolor': ink,
                         'axes.prop_cycle': plt.cycler(color=[color, '#d6883b', '#635aa6'])})
    pages = []
    pdf = PdfPages(out/'LPSim-full-report.pdf', metadata={
        'Title': 'LPSim | Berkeley traffic and computational performance',
        'Author': 'LPSim report generator', 'Subject': 'Full recorded-run analysis; observed coverage'})

    def page(title, subtitle):
        fig = plt.figure(figsize=(11.7, 8.3), facecolor='white')
        fig.text(.065, .95, 'LPSIM   /   BERKELEY RECORDED RUN', color=color, fontsize=10, weight='bold')
        fig.text(.065, .89, title, fontsize=23, weight='bold')
        fig.text(.065, .85, subtitle, fontsize=10, color=muted)
        return fig

    def finish(fig, caption):
        fig.text(.065, .065, textwrap.fill(caption, 143), fontsize=8.3, color=muted, va='top')
        number = len(pages)+1
        fig.text(.065, .022, 'FULL RECORDED DATA  |  Observed metrics; see definitions and coverage', fontsize=8, color=muted)
        fig.text(.935, .022, f'{number:02d}', ha='right', fontsize=9, color=color)
        name = f'page-{number:02d}.png'
        fig.savefig(out/name, dpi=150)
        pdf.savefig(fig)
        plt.close(fig)
        pages.append(name)

    def table(fig, headers, rows, box=(.065, .15, .87, .64), widths=None, size=10):
        ax = fig.add_axes(box); ax.axis('off')
        t = ax.table(cellText=rows, colLabels=headers, cellLoc='left', colLoc='left',
                     colWidths=widths, bbox=[0, 0, 1, 1])
        t.auto_set_font_size(False); t.set_fontsize(size)
        for (r,c), cell in t.get_celld().items():
            cell.set_edgecolor('white'); cell.PAD = .08
            if r == 0:
                cell.set_facecolor(ink); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
            else: cell.set_facecolor('#eef4f6' if r % 2 else '#f8fafb')

    def axes_grid(fig, rows=2, cols=2):
        return fig.subplots(rows, cols, squeeze=False, gridspec_kw={
            'left': .09, 'right': .94, 'bottom': .17, 'top': .77, 'hspace': .56, 'wspace': .30})

    def values(rows, key, scale=1):
        return np.array([r[key]*scale for r in rows if np.isfinite(r.get(key,np.nan))])

    start = min(v['first_sample_s'] for v in vehicles)
    end = max(v['last_sample_s'] for v in vehicles)
    fig = page('Traffic & computational performance', 'Full recorded dataset  •  Berkeley network  •  Offline analysis')
    cards = [('OBSERVED VEHICLES', f"{summary['vehicles_observed']:,}"),
             ('OBSERVED ROAD SEGMENTS', f"{summary['segments_observed']:,}"),
             ('VEHICLE-MILES', f"{summary['observed_vmt']:,.1f}"),
             ('VEHICLE-HOURS', f"{summary['observed_vht']:,.1f}")]
    for i,(label,val) in enumerate(cards):
        x = .07 + (i%2)*.46; y = .70-(i//2)*.17
        fig.text(x,y,label,fontsize=10,color=muted)
        fig.text(x,y-.065,val,fontsize=31,weight='bold',color=color)
    speed = summary['observed_vmt'] / summary['observed_vht']
    overview = (f"{summary['quality']['input_samples']:,} trajectory samples; simulation clock {start/3600:.2f}–{end/3600:.2f} hours.\n"
                f"Network distance/time speed: {speed:.2f} mph ({speed*.44704:.2f} m/s).\n"
                'Traffic totals cover all valid observations in the supplied recording.\n'
                'GPU utilization, warp activity, step timing and partition telemetry were not recorded.')
    fig.text(.07,.31,overview,fontsize=12,linespacing=1.9,va='top')
    finish(fig,'This is the full recording, not the earlier 100,000-row sample. Complete trip travel times require insertion and completion events.')

    fig = page('How to read the metrics', 'Definitions and assumptions apply to every figure and summary table.')
    table(fig,['Metric','Definition / interpretation'],[
        ['VMT','Valid route-distance increments / 1,609.344; vehicle-miles.'],
        ['VHT','Valid inter-sample vehicle time / 3,600; vehicle-hours.'],
        ['Distance/time speed','Total distance / total vehicle time; includes stopped time.'],
        ['Observed vehicle duration','Last minus first sample; excludes unobserved trip endpoints.'],
        ['Road travel-time proxy','Road length / aggregate road speed; not measured mean traversal time.'],
        ['Estimated road traversal','Both boundary crossings observed; crossing times interpolated by distance.'],
        ['Stopped time','Valid interval time with reconstructed speed below 0.1 m/s.'],
        ['Estimated delay','Time above road speed-limit baseline; includes multiple causes of delay.'],
        ['Time axes','Traffic uses simulation time. Compute telemetry requires its own time axis.'],
        ['Missing compute data','Unavailable, never zero. GPU utilization and warp occupancy are distinct.']
    ], widths=[.26,.74], size=10)
    finish(fig,'Road-boundary times are allocated proportionally to distance within a sampling interval. Invalid gaps are excluded from VMT and VHT.')

    for title, rows, specs, subtitle in [
        ('Vehicle metric summary',vehicles,[('observed_span_s','Observed duration (s)'),('valid_observed_time_s','Valid observed time (s)'),('observed_vmt','VMT (vehicle-miles)'),('observed_vht','VHT (vehicle-hours)'),('distance_time_speed_mps','Distance/time speed (m/s)'),('mean_sampled_speed_mps','Mean sampled speed (m/s)'),('observed_stopped_s','Stopped time (s)'),('estimated_free_flow_delay_s','Estimated delay (s)')], 'Each vehicle has equal weight in the percentiles below.'),
        ('Road segment metric summary',roads,[('observed_vmt','VMT (vehicle-miles)'),('observed_vht','VHT (vehicle-hours)'),('distance_time_speed_mps','Distance/time speed (m/s)'),('length_over_mean_speed_s','Travel-time proxy (s)'),('mean_estimated_traversal_s','Estimated mean traversal (s)'),('estimated_complete_traversals','Estimated traversal count'),('observed_stopped_s','Stopped vehicle-seconds'),('estimated_free_flow_delay_s','Estimated delay (vehicle-s)')], 'Each observed road segment has equal weight; traversal estimates exclude roads without both crossings.')]:
        fig=page(title,subtitle)
        stats=[]
        for key,label in specs:
            v=values(rows,key)
            stats.append([label,f'{len(v):,}']+[f'{x:,.2f}' for x in [np.mean(v),*np.percentile(v,[50,90,95])]] if len(v) else [label,'0','—','—','—','—'])
        table(fig,['Metric','Count','Mean','Median','P90','P95'],stats,widths=[.36,.10,.135,.135,.135,.135],size=10)
        finish(fig,'Mean road traversal is first calculated per road, then summarized across roads. Vehicle duration is observed coverage, not completed trip time.')

    fig=page('Vehicle performance distributions','Duration, distance, speed and stopped time across all observed vehicles.')
    specs=[('observed_span_s','Observed duration (minutes)',1/60),('observed_vmt','Observed distance (miles)',1),('distance_time_speed_mps','Distance/time speed (m/s)',1),('observed_stopped_s','Stopped time (minutes)',1/60)]
    for ax,(key,label,scale) in zip(axes_grid(fig).flat,specs):
        ax.hist(values(vehicles,key,scale),bins=45,color=color);ax.set_xlabel(label);ax.set_ylabel('Vehicles');ax.grid(axis='y',alpha=.18)
    finish(fig,'All valid vehicles are included. Histograms retain the full range; no tails are removed. Stopped time uses reconstructed interval speed.')

    fig=page('Network performance over time',f"{summary['bin_seconds']:g}-second bins on the simulation clock; partial bins are included.")
    x=np.array([b['simulation_time_s']/3600 for b in bins])
    for ax,(key,label) in zip(axes_grid(fig).flat,[('distance_time_speed_mps','Network speed (m/s)'),('mean_observed_vehicles','Mean observed vehicles'),('observed_vmt','VMT per bin'),('observed_vht','VHT per bin')]):
        ax.plot(x,[b[key] for b in bins]);ax.set_ylabel(label);ax.set_xlabel('Simulation clock (hours)');ax.grid(alpha=.18)
    finish(fig,'Partial first/last bins contain less observation time. Mean observed vehicles uses each bin’s recorded temporal span, not necessarily its full width.')

    fig=page('Accumulated travel and time','VMT and VHT accumulated across the complete recording.')
    for ax,(key,label) in zip(axes_grid(fig,2,1).flat,[('observed_vmt','Cumulative vehicle-miles'),('observed_vht','Cumulative vehicle-hours')]):
        ax.plot(x+summary['bin_seconds']/3600,np.cumsum([b[key] for b in bins]));ax.set_ylabel(label);ax.set_xlabel('Simulation clock (hours; bin end)');ax.grid(alpha=.18)
    finish(fig,'Each cumulative point includes that entire bin’s recorded observations. The final point contains the complete observed-run total.')

    for title,specs in [('Road speed and vehicle-hours',[('distance_time_speed_mps','Speed (m/s)'),('observed_vht','VHT (vehicle-hours)')]),('Road distance and travel time',[('observed_vmt','VMT (vehicle-miles)'),('length_over_mean_speed_s','Travel-time proxy (s)')])]:
        fig=page(title,'Spatial distribution across observed road segments; unobserved roads are omitted.')
        for ax,(key,label) in zip(axes_grid(fig,1,2).flat,specs):
            geom=[edges[int(r['edge_id'])].points for r in roads]
            lc=LineCollection(geom,array=np.array([r[key] for r in roads]),cmap='viridis',linewidths=.9)
            ax.add_collection(lc);ax.autoscale();ax.set_aspect(1/np.cos(np.deg2rad(37.87)))
            ax.xaxis.set_major_locator(MaxNLocator(3));ax.tick_params(labelsize=8)
            ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.set_title(label)
            fig.colorbar(lc,ax=ax,shrink=.7,pad=.03)
        finish(fig,'Travel-time proxy = road length / aggregate speed. Missing proxies are omitted. Recorded network geometry is used; endpoint lines fill missing curves.')

    fig=page('Roads with the highest estimated delay','Accumulated excess vehicle-hours relative to each road’s speed-limit baseline.')
    ranked=sorted(roads,key=lambda r:r['estimated_free_flow_delay_s'],reverse=True)[:12]
    table(fig,['Road ID','VMT','VHT','Speed m/s','Delay veh-h','Traversal s'],[
        [str(int(r['edge_id'])),f"{r['observed_vmt']:,.1f}",f"{r['observed_vht']:,.2f}",f"{r['distance_time_speed_mps']:.2f}",f"{r['estimated_free_flow_delay_s']/3600:.2f}",f"{r['mean_estimated_traversal_s']:.1f}" if np.isfinite(r['mean_estimated_traversal_s']) else 'Unavailable'] for r in ranked
    ],size=10)
    finish(fig,'High traffic volumes increase total delay. This ranking does not isolate signal delay or rank only by congestion. Traversal seconds use interpolated complete crossings.')

    fig=page('Computational performance coverage','No matching computational telemetry was supplied for this historical recording.')
    table(fig,['Requested measurement','Status','Required source'],[
        ['GPU utilization over time','Unavailable','Timestamped device telemetry'],
        ['Warp activity / occupancy','Unavailable','Timestamped profiler counters'],
        ['Eligible / active warps','Unavailable','Profiler export with counter definitions'],
        ['Partition load / balance','Unavailable','Per-partition active-vehicle counts'],
        ['Partition migrations / transfers','Unavailable','Migration and transfer-byte counters'],
        ['Step duration / throughput','Unavailable','Step timing + simulation clock'],
        ['GPU temperature / power / RAM','Unavailable','Matching device telemetry'],
        ['Whole-command wall runtime','Unavailable','Matching recorder capture metadata']
    ],widths=[.37,.17,.46],size=10)
    finish(fig,'These values cannot be reconstructed from traffic trajectories. No synthetic measurements are substituted. A new instrumented run is required for a fully populated compute section.')

    fig=page('Coverage, validation and provenance','Tables and original run inputs were checked against the saved SHA-256 manifest.')
    quality=summary['quality']
    rows=[['Input samples',f"{quality['input_samples']:,}"],['Accepted samples',f"{quality['accepted_samples']:,}"],['Duplicate samples',f"{quality.get('duplicates',0):,}"],['Invalid samples',f"{quality.get('invalid_samples',0):,}"],['Conflicting timestamps',f"{quality.get('conflicting_timestamps',0):,}"],['Invalid intervals',f"{quality.get('invalid_intervals',0):,}"],['Vehicles missing routes',f"{quality.get('vehicles_missing_routes',0):,}"],['Source tables',str(src.relative_to(Path(__file__).resolve().parents[1]))],['Trajectory SHA-256',next(h[:32]+'…' for p,h in manifest['inputs'].items() if p.endswith('/trajectories.csv'))]]
    table(fig,['Validation item','Result'],rows,widths=[.34,.66],size=10)
    finish(fig,'Checksums verify input identity, not simulation correctness. Historical build, random seed, full trip endpoints, and missing compute measurements are not inferred.')
    pdf.close()
    encoded=base64.b64encode((out/'LPSim-full-report.pdf').read_bytes()).decode()
    images=''.join(f'<section id="page-{i}"><h2>Page {i} of {len(pages)}</h2><img alt="Report page {i}" src="data:image/png;base64,{base64.b64encode((out/name).read_bytes()).decode()}"></section>' for i,name in enumerate(pages,1))
    (out/'read-report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>LPSim full report</title><style>body{margin:0;background:#edf2f5;color:#17324d;font:16px system-ui}header{padding:18px 5%;background:white;position:sticky;top:0;box-shadow:0 2px 8px #0002}a{color:#087e8b;margin-right:25px}main{max-width:1200px;margin:auto}section{margin:30px 0}img{width:100%;box-shadow:0 2px 12px #0002}h2{font-size:15px} @media print{header,h2{display:none}section{break-after:page;margin:0}}</style><header><strong>Berkeley · Full recorded-run report</strong><p><a download="LPSim-full-report.pdf" href="data:application/pdf;base64,'+encoded+'">Download full PDF</a><span>'+str(len(pages))+' pages · 20,000 vehicles · missing compute data labeled</span></p></header><main>'+images+'</main></html>')
    output_manifest={'source_manifest':str(src/'manifest.json'),'source_manifest_sha256':digest(src/'manifest.json'),'input_samples':quality['input_samples'],'pages':len(pages),'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    (out/'manifest.json').write_text(json.dumps(output_manifest,indent=2))
    print(f'Ready: {out / "LPSim-full-report.pdf"} ({len(pages)} pages)',flush=True)


if __name__ == '__main__':
    main()
