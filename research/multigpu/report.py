"""Generate the preparation report and figures from actual validation outputs."""
from pathlib import Path
import argparse, csv, json, statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.bundle/'validation_results.json').read_text());gate=json.loads((a.bundle/'validation_gate.json').read_text())
selected=[r for r in rows if r['case']=='reference_2048_t0' and r['index']=='bulk']
summary=[]
for r in selected:
    with (a.bundle/'validation'/r['name']/'steps.csv').open() as f:steps=list(csv.DictReader(f))
    mean_active=[statistics.mean(int(s[f'partition{i}_active']) for s in steps) for i in range(3)]
    meta=json.loads((a.bundle/'inputs'/r['case']/(r['policy']+'.json')).read_text());m=r['metrics']
    summary.append(dict(policy=r['policy'],cut_roads=meta['cut_roads'],cut_lanes=meta['cut_lane_count'],transfers=m['ownership_transfers'],state_messages=m['state_messages'],reservation_messages=m['reservation_messages'],wire_mb=m['wire_bytes']/1e6,owner_map_mb=m['actual_owner_map_h2d_bytes']/1e6,mean_active_0=mean_active[0],mean_active_1=mean_active[1],mean_active_2=mean_active[2],completed=m['completed']))
with (a.output/'policy-summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
table='| Policy | Cut roads | Ownership transfers | State + reservation payload (MB) | Completed |\n|---|---:|---:|---:|---:|\n'
for r in summary:table+=f"| {r['policy']} | {r['cut_roads']:,} | {r['transfers']:,} | {r['wire_mb']:.2f} | {r['completed']} |\n"
text=f'''# Three-partition preparation results — October 27, 2026

**Correctness gate passed:** {gate['runs']} partitioned runs, {gate['compared_timesteps']:,} timestep comparisons; every vehicle record was bitwise equal to the original compact/radix reference at every step. Road, connector, cross-boundary gap and conflicting-movement checks passed. Policies produced identical physical results. The undersized-message-pool test failed closed as required.

Full 600 s tests used the historical 2,048-trip distributed demand and a fresh held-out 2,048-trip bursty demand, each with all three policies. Populated 8,192- and 16,384-trip snapshots exercised higher active populations over 120 and 40 steps respectively. These are **not** full-horizon high-population runs. Additional runs checked incremental indexing and all-red/all-permitted controls.

The four historical full runs (three bulk policies plus incremental geography) each matched all 19 available nonzero historical 30 s checkpoints. Standalone bulk and incremental builds additionally matched all 20 archived snapshots (including t=0), and their trajectory/event/population outputs matched each other. Incremental indexing remains separately selectable; the bulk reference is unchanged.

## Communication tradeoff on identical traffic

Historical 2,048 trips, 600 s, mode 0, bulk index:

{table}
The vehicle-count baseline initially sees no active vehicles and explicitly falls back to unit junction weights. Its scattered junction assignment cuts many roads. Therefore this table demonstrates communication consequences of the policies, **not a fair proof against online active-count balancing**. Populated-snapshot tests use actual active counts and their outputs are retained separately.

The graph heuristic improves communication relative to scattered count assignment but has more boundary traffic than contiguous geography in this case. Geography is a strong baseline a learned prior must beat. None of these results demonstrates a benefit from forecasting.

The payload column counts each state/reservation message once. Host staging copies it D2H and H2D, so it is not measured inter-GPU link traffic. Each policy additionally copied {summary[0]['owner_map_mb']:.2f} MB of full ownership maps H2D over this run. Verification also copies all partition/reference vehicle arrays; those copies are outside partition execution timing and are not in the payload column. Dense stable-ID arrays allocate N slots per partition plus replicated routes/network metadata.

## What is ready

- A source/input/output-hashed reference bundle, rebuild commands and recorded Jetson hardware/power configuration.
- A complete ownership and stage-dependency protocol, independent partition buffers, persistent message pools, deterministic transfers and capacity guards.
- Three policies behind the same ownership-map interface, with policy metadata and observed population counts per timestep.
- Frozen training/validation/test demand seeds 2701/2702/2703: 27 cases across three patterns and three requested populations.
- A digest-pinned CUDA container recipe, Modal SDK 1.6.3 runner, sm_80 build, hardware/peer-transfer probe, profiler-permission probe and automatic artifact collection.

## Limits and next gate

Jetson partitions execute sequentially with host-staged messages. These results establish correctness and communication volume; they do not establish A100 throughput or speedup. No remote Modal build or GPU launch has occurred. SDK definitions were imported locally; actual hardware, remote compilation and profiler permissions remain untested.

The fixed traffic model remains synthetic: 72 s signals, inferred lane permissions, route-assigned lanes, no pedestrians or calibrated OD demand. There is no larger validated network locally, and the fresh splits share Berkeley topology. Cross-network generalization remains future work.

After A100 access: run the prepared preflight, add and validate concurrent peer transport, then run separate timing repetitions including all allocation and communication costs. Test an oracle before training. The preregistered success gate is a 10% geometric-mean complete-runtime reduction over the strongest baseline on held-out demand, a positive paired confidence bound, no case regressing more than 5%, identical physics and the same memory budget.

Raw results: `validation_results.json`, `validation_gate.json`, each run's `steps.csv`, `metrics.json`, logs and snapshots in the artifact bundle. Protocol and evaluation rules are in `research/multigpu/protocol.md` and `evaluation.md`.
'''
(a.output/'README.md').write_text(text)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
with PdfPages(a.output/'multigpu-preparation.pdf') as pdf:
    fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.92,'Three-partition preparation: correctness established',fontsize=19,weight='bold');fig.text(.07,.83,f"{gate['runs']} runs  |  {gate['compared_timesteps']:,} timestep comparisons  |  zero mismatches",fontsize=15)
    cells=[[r['policy'],f"{r['cut_roads']:,}",f"{r['transfers']:,}",f"{r['wire_mb']:.2f}",str(r['completed'])] for r in summary]
    ax=fig.add_axes([.07,.40,.86,.32]);ax.axis('off');tab=ax.table(cellText=cells,colLabels=['Policy','Cut roads','Transfers','Payload MB','Completed'],loc='center',cellLoc='center',colWidths=[.30,.16,.19,.19,.16]);tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.1)
    fig.text(.07,.33,'Identical 2,048-trip, 600 s traffic. Payload counts messages once; host copies cost more.',fontsize=11)
    fig.text(.07,.24,'Vehicle-count policy uses unit-junction fallback at empty start.\nThis is a communication tradeoff, not evidence against adaptive active-count balancing.\nNo A100 performance or learned-prior benefit has been demonstrated.',fontsize=12,linespacing=1.6)
    fig.text(.07,.07,'LPSim • 2026-10-27 • Jetson correctness backend; no new Nsight performance claims',color='#526879');pdf.savefig(fig);plt.close(fig)
    geo=json.loads((a.bundle/'inputs/network/geometry.json').read_text());segments=[[r['start'],r['end']] for r in geo['roads']]
    fig,axes=plt.subplots(1,3,figsize=(12,7))
    for ax,r in zip(axes,summary):
        owners=[int(x) for x in (a.bundle/'inputs/reference_2048_t0'/(r['policy']+'.txt')).read_text().split()]
        colors=[['#227c9d','#e7a33e','#b54c69'][owners[e['b']]] for e in geo['roads']]
        ax.add_collection(LineCollection(segments,colors=colors,linewidths=.5));ax.autoscale();ax.set_aspect('equal');ax.set_title(r['policy']);ax.axis('off')
    fig.suptitle('Same Berkeley network; different ownership boundaries',fontsize=16);fig.text(.08,.04,'Road color follows its downstream junction owner. © OpenStreetMap contributors. Synthetic research geometry.',fontsize=9);fig.savefig(a.output/'partitions.png',dpi=170,bbox_inches='tight');pdf.savefig(fig);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11.7,7));labels=['Geography','Count\n(empty-start fallback)','Graph heuristic']
    axes[0].bar(labels,[r['wire_mb'] for r in summary],color=['#227c9d','#e7a33e','#b54c69']);axes[0].set_ylabel('Message payload MB / 600 s');axes[0].set_title('Boundary message volume')
    for i in range(3):axes[1].bar([j+i*.23-.23 for j in range(3)],[r[f'mean_active_{i}'] for r in summary],width=.23,label=f'Partition {i}')
    axes[1].set_xticks(range(3),labels);axes[1].set_ylabel('Mean road + connector population');axes[1].set_title('Active population balance');axes[1].legend()
    fig.suptitle('Communication and active balance must be measured together',fontsize=15);fig.tight_layout(rect=[0,.05,1,.94]);fig.text(.05,.02,'Payload excludes owner-map broadcasts and validation copies. These are not multi-GPU timing results.',fontsize=10);fig.savefig(a.output/'communication-and-balance.png',dpi=170);pdf.savefig(fig);plt.close(fig)
    fig=plt.figure(figsize=(11.7,8.3));fig.text(.08,.91,'Ready now / remaining hardware gate',fontsize=20,weight='bold')
    lines=['READY: original reference + selectable incremental indexing, hashes and rebuild commands.', 'READY: stage-specific boundary protocol, separate buffers, message packing and capacity checks.', 'READY: three partition policies; train/validation/test inputs and preregistered performance gate.', 'READY: pinned Modal/CUDA recipe, hardware discovery, peer-copy and counter-permission probes.', '', 'NOT YET: actual A100 validation, remote build, peer topology or profiler permissions.', 'NOT YET: concurrent GPU execution, sparse local storage, direct peer transport or scaling results.', '', 'Next: preflight → concurrent transport → measured oracle → held-out prior experiment.', 'Success: ≥10% complete-runtime improvement over strongest baseline, charging all overhead.', 'Do not optimize for utilization alone; correctness and net wall time remain the criteria.']
    fig.text(.08,.79,'\n\n'.join(lines),fontsize=11,va='top');pdf.savefig(fig);plt.close(fig)
print(a.output/'multigpu-preparation.pdf')
