"""Verify returned artifacts and produce tables, figures and a standalone PDF."""
from pathlib import Path
import csv,hashlib,io,json,tarfile,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/exchange-study-20261029';OUT=ROOT/'reports/exchange-study-20261029'
OUT.mkdir(exist_ok=True,parents=True)
archive=next((ART/'cloud').glob('*.tar.gz'));raw=ART/'raw';raw.mkdir(exist_ok=True)
with tarfile.open(archive) as t:
    for m in t.getmembers():
        if m.issym() or m.islnk() or not (raw/m.name).resolve().is_relative_to(raw.resolve()):raise ValueError('Unsafe archive entry')
    t.extractall(raw)
folder=next(raw.glob('exchange-*'))
for name,digest in json.loads((folder/'sha256.json').read_text()).items():
    assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
status=json.loads((folder/'status.json').read_text())
if not status['passing']:raise RuntimeError(status)
runs=pd.DataFrame(json.loads((folder/'runs.json').read_text()))
assert all(runs.groupby(['case','steps']).final_sha256.nunique()==1)
chosen=json.loads((folder/'backend-selection.json').read_text())['chosen']
tests=runs[runs.phase=='test'].copy()
keys=['case','backend','policy','ranks','mode']
timing=tests.groupby(keys).agg(median_ms=('compute_ms','median'),min_ms=('compute_ms','min'),max_ms=('compute_ms','max'),repeats=('compute_ms','size'),setup_ms=('setup_ms','median'),wire_bytes=('wire_bytes','median')).reset_index()
assert all(timing.repeats==5)
runs.to_csv(OUT/'all-runs.csv',index=False);timing.to_csv(OUT/'timing-summary.csv',index=False)
traces=[];steps=[];pop=[]
for row in runs.to_dict('records'):
    meta={k:row[k] for k in keys+['phase','repeat']}
    if row['diagnostic']:
        d=pd.read_csv(folder/row['path']/'trace.csv');d['host_call_ms']=(d.host_end_ns-d.host_begin_ns)/1e6
        traces.append(d.assign(**meta))
    if row['phase']=='test':
        steps.append(pd.read_csv(folder/row['path']/'steps.csv').assign(**meta))
    if (folder/row['path']/'populations.csv').exists():
        pop.append(pd.read_csv(folder/row['path']/'populations.csv').assign(**meta))
trace=pd.concat(traces,ignore_index=True);trace.to_csv(OUT/'per-gpu-trace.csv',index=False)
stage=pd.concat(steps,ignore_index=True);stage.to_csv(OUT/'steps.csv',index=False)
pop=pd.concat(pop,ignore_index=True);pop.to_csv(OUT/'populations.csv',index=False)
alloc=[]
for key,g in pop[(pop.phase=='test')&(pop.backend=='new')&(pop['mode']==chosen)].groupby(['case','policy','ranks','repeat']):
    last=g[g.step==g.step.max()];active=last.stopped+last.moving+last.connector
    alloc.append(dict(case=key[0],policy=key[1],ranks=key[2],repeat=key[3],
        active=float(active.sum()),active_max_mean=float(active.max()/active.mean()),
        ghosts=float(last.ghosts.sum()),processed_records=float(last.processed_records.sum()),
        reserved_capacity=float(last.reserved_capacity.sum()),pending=float(last.pending.sum())))
allocation=pd.DataFrame(alloc).groupby(['case','policy','ranks']).mean(numeric_only=True).reset_index().drop(columns='repeat')
allocation.to_csv(OUT/'allocation-summary.csv',index=False)
summary=trace.groupby(['case','mode','rank','stage','component']).agg(gpu_median_ms=('gpu_ms','median'),host_call_median_ms=('host_call_ms','median')).reset_index()
summary.to_csv(OUT/'exchange-components.csv',index=False)
wait=trace[trace.component=='host_device_sync'].copy()
wait['last_rank_finish_ns']=wait.groupby(['case','mode','step','stage']).host_end_ns.transform('max')
wait['finish_spread_ms']=(wait.last_rank_finish_ns-wait.host_end_ns)/1e6
wait.to_csv(OUT/'rank-wait-proxies.csv',index=False)
gpu_rank=(trace[trace.component!='host_device_sync']
          .groupby(['case','mode','rank','step','component']).gpu_ms.sum()
          .groupby(['case','mode','rank','component']).mean().unstack('component').reset_index())
gpu_rank.to_csv(OUT/'per-gpu-component-means.csv',index=False)

policies=['geography','paper_balanced','paper_community','work_balanced','boundary_aware']
def construction(case,policy,k,include_feature=True):
    m=json.loads((ART/'inputs'/case/f'{policy}_{k}.json').read_text())
    return m.get('total_construction_ms',m['construction_ms']) if include_feature else m['construction_ms']

selector=[]
for case in sorted(runs.case.unique()):
    selection=json.loads((folder/(case+'-selection.json')).read_text())
    for k in (2,3):
        selected=selection['measured_cost_candidate'][str(k)]['policy']
        calibration=runs[(runs.case==case)&(runs.phase=='policy_train')&(runs.ranks==k)]
        means=calibration.groupby('policy').compute_ms.median()
        base=min(policies[:3],key=lambda p:(means[p],p))
        def test(p):return tests[(tests.case==case)&(tests.backend=='new')&(tests.ranks==k)&(tests['mode']==chosen)&(tests.policy==p)].sort_values('repeat')
        b,c=test(base),test(selected)
        # Bootstrap the SAME statistic reported below: ratio of test medians.
        # Preserve each randomized repetition block when resampling both policies.
        bv,cv=b.compute_ms.to_numpy(),c.compute_ms.to_numpy()
        ix=np.random.default_rng(2910).integers(0,len(bv),(10000,len(bv)))
        boot=100*(1-np.median(cv[ix],axis=1)/np.median(bv[ix],axis=1))
        construction_ms=sum(construction(case,p,k,False) for p in policies)
        feature=json.loads((ART/'inputs'/case/'work_balanced_2.json').read_text())['feature_construction_ms']
        construction_ms+=feature # Shared new-policy feature extraction charged once.
        calibration_ms=1000*calibration.process_seconds.sum()
        cold_baseline=float(b.compute_ms.median()+b.setup_ms.median()+construction(case,base,k))
        charged=float(c.compute_ms.median()+c.setup_ms.median()+construction_ms+calibration_ms+selection['selection_cpu_ms'])
        selector.append(dict(case=case,ranks=k,selected=selected,baseline_selected_from_train=base,
            baseline_test_ms=b.compute_ms.median(),selected_test_ms=c.compute_ms.median(),
            optimistic_gain_pct=100*(1-c.compute_ms.median()/b.compute_ms.median()),
            repeat_bootstrap_low_pct=np.quantile(boot,.025),repeat_bootstrap_high_pct=np.quantile(boot,.975),
            calibration_process_ms=calibration_ms,all_map_construction_ms=construction_ms,
            baseline_cold_ms=cold_baseline,charged_selector_ms=charged,
            net_cold_gain_pct=100*(1-charged/cold_baseline),heldout_demand=False))
selector=pd.DataFrame(selector);selector.to_csv(OUT/'selector.csv',index=False)

nsight=[]
for p in folder.glob('profile-mode*/metrics.csv'):
    lines=p.read_text().splitlines();head=next((i for i,l in enumerate(lines) if l.startswith('"ID"')),None)
    if head is None:continue
    parsed=list(csv.reader(io.StringIO('\n'.join(lines[head:]))));columns=parsed[0]
    if 'Metric Name' in columns:
        rows=[dict(zip(columns,x)) for x in parsed[1:]]
    else:
        units=parsed[1];rows=[]
        for values in parsed[2:]:
            r=dict(zip(columns,values))
            if not r.get('ID','').isdigit():continue
            for i,name in enumerate(columns):
                if '__' in name:
                    rows.append({**{k:r.get(k,'') for k in ['ID','Kernel Name','Device','Context','Stream']},'Metric Name':name,'Metric Unit':units[i],'Metric Value':r.get(name,'')})
    for r in rows:
        if not r.get('Metric Name'):continue
        try:value=float(r['Metric Value'].replace(',',''))
        except (ValueError,KeyError):continue
        kernel=r.get('Kernel Name','');family=next((x for x in ['advanceCompact','arbitrateIndexedJunctions','packLocal','packSlots','gatherLocal','gatherSlots'] if x in kernel),kernel)
        nsight.append(dict(mode=int(p.parent.name.replace('profile-mode','')),family=family,value=value,**r))
nsight=pd.DataFrame(nsight);nsight.to_csv(OUT/'nsight-raw.csv',index=False)
if len(nsight):
    ns=nsight.groupby(['mode','family','Metric Name','Metric Unit']).value.median().reset_index()
    ns.to_csv(OUT/'nsight-summary.csv',index=False)
else:ns=pd.DataFrame()

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
figures=[]
def save(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),dpi=170);fig.savefig(OUT/(name+'.pdf'));figures.append((name,fig))
cases=sorted(runs.case.unique());labels=['Original exchange','Persistent sort','Slot retention','Combined']
fig,axes=plt.subplots(1,len(cases),figsize=(12,4),squeeze=False)
for ax,case in zip(axes[0],cases):
    d=timing[(timing.case==case)&(timing.backend=='new')&(timing.policy=='paper_balanced')&(timing.ranks==3)].sort_values('mode')
    ax.bar(d['mode'],d.median_ms,color=['#7793a8','#57a38b','#dda75b','#547cad'])
    ax.set_xticks(range(4),labels,rotation=25,ha='right');ax.set_title(case);ax.set_ylabel('Wall ms / 60 steps; five test repetitions')
save(fig,'01-framework-ablation')
fig,axes=plt.subplots(len(cases),2,figsize=(12,7),squeeze=False)
for i,case in enumerate(cases):
    for j,k in enumerate((2,3)):
        d=timing[(timing.case==case)&(timing.backend=='new')&(timing['mode']==chosen)&(timing.ranks==k)].set_index('policy').loc[policies]
        ax=axes[i,j];ax.bar(range(5),d.median_ms,color='#547cad');ax.set_xticks(range(5),['Geography','Paper balance','Paper community','Work balance','Boundary aware'],rotation=25,ha='right');ax.set_title(f'{case}, {k} GPUs, common mode {chosen}');ax.set_ylabel('Wall ms / 60 steps')
save(fig,'02-policy-comparison')
components=summary[(summary.case=='sf_98304')&(summary.component!='host_device_sync')].copy()
components['name']=components.stage+'.'+components.component
fig,axes=plt.subplots(1,2,figsize=(13,6))
for ax,field,title in zip(axes,['gpu_median_ms','host_call_median_ms'],['GPU event spans (may include submission gaps)','Host call spans (includes blocking API work)']):
    d=components[components['mode'].isin(sorted({0,chosen}))].groupby(['name','mode'])[field].mean().unstack('mode').fillna(0)
    d=d.loc[d.max(axis=1).nlargest(12).index].sort_values(by=d.columns[0]);d.plot.barh(ax=ax,width=.8);ax.set_title(title);ax.set_xlabel('ms; mean of per-rank medians');ax.set_ylabel('')
save(fig,'03-exchange-costs')
fig,ax=plt.subplots(figsize=(11,4));x=np.arange(len(selector));w=.25
ax.bar(x-w,selector.baseline_test_ms,w,label='Baseline simulation only');ax.bar(x,selector.selected_test_ms,w,label='Pre-supplied measured-cost choice');ax.bar(x+w,selector.charged_selector_ms,w,label='Choice + calibration/setup/construction')
ax.set_yscale('log');ax.set_xticks(x,[f'{r.case}\n{r.ranks} GPUs' for r in selector.itertuples()]);ax.set_ylabel('ms, logarithmic scale');ax.legend(fontsize=8);save(fig,'04-selector-costs')

def page(pdf,title,paragraphs):
    fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,title,fontsize=18,weight='bold');y=.85
    for p in paragraphs:
        lines=textwrap.wrap(p,115)
        fig.text(.06,y,'\n'.join(lines),fontsize=11,va='top',linespacing=1.5);y-=.035*len(lines)+.03
    pdf.savefig(fig);plt.close(fig)
def table(pdf,title,df,footer=''):
    fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,fontsize=16,pad=20)
    d=df.copy()
    for col in d:
        if pd.api.types.is_float_dtype(d[col]):d[col]=d[col].map(lambda x:f'{x:.3f}')
    tab=ax.table(cellText=d.values,colLabels=d.columns,loc='center',cellLoc='center');tab.auto_set_font_size(False);tab.set_fontsize(8);tab.scale(1,1.65)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('#d6dfe4');cell.set_facecolor('#deebef' if r==0 else ('#f2f5f7' if r%2 else 'white'))
    fig.text(.06,.04,footer,fontsize=9,wrap=True);pdf.savefig(fig);plt.close(fig)

valid_steps=int(runs[runs.verify].steps.sum())
with PdfPages(OUT/'exchange-policy-report.pdf') as pdf:
    page(pdf,'Exchange costs and allocation policies — October 29, 2026',[
        f'Completed one bounded three-A100 study: {len(tests)} independent test timing runs, {valid_steps} exact-reference timestep comparisons. Training selected exchange mode {chosen} ({labels[chosen]}). All retained physical hashes agree for each workload/horizon.',
        'Mechanisms: persistent radix-sort workspace and slot-reference retention were tested separately and combined. Full-record peer payload, physics, global priority, authoritative reservations and stage barriers remain unchanged. Every allocation policy uses the same selected backend.',
        'Workloads: Berkeley 8,192 and SF 98,304 initially active vehicles, 60 original 0.25-s steps. These constructed active snapshots test workload behavior, not calibrated demand. Independent timing repetitions are not held-out traffic patterns.',
        'The measured-cost choice is selected using three training repetitions and evaluated on five separate shuffled repetitions. Its optimistic benefit and its calibration/construction/setup charges are reported separately. No learned predictor was trained.',
        'Diagnostics use a separate compiled binary. GPU event spans can include host submission gaps; host synchronization spans include pending device work. Hardware-counter replay is separate from all timing results. See raw CSVs for per-GPU values.' ])
    framework=[];framework_rows=[]
    for case in cases:
        x=timing[timing.case==case];old=x[x.backend=='legacy'].median_ms.iloc[0]
        new=x[(x.backend=='new')&(x['mode']==chosen)&(x.ranks==3)&(x.policy=='paper_balanced')].median_ms.iloc[0]
        framework.append(f'{case}: {old:.3f} to {new:.3f} ms / 60 steps ({100*(1-new/old):.1f}% faster than the unchanged local backend).')
        q=tests[(tests.case==case)&(tests.ranks==3)&(tests.policy=='paper_balanced')]
        bv=q[q.backend=='legacy'].sort_values('repeat').compute_ms.to_numpy()
        cv=q[(q.backend=='new')&(q['mode']==chosen)].sort_values('repeat').compute_ms.to_numpy()
        ix=np.random.default_rng(2910).integers(0,len(bv),(10000,len(bv)))
        boot=100*(1-np.median(cv[ix],axis=1)/np.median(bv[ix],axis=1))
        framework_rows.append(dict(case=case,gain_pct=100*(1-new/old),low_pct=np.quantile(boot,.025),high_pct=np.quantile(boot,.975)))
    pd.DataFrame(framework_rows).to_csv(OUT/'framework-comparison.csv',index=False)
    page(pdf,'Findings: framework gain, limited policy opportunity',[
        'Framework: '+' '.join(framework),
        'SF diagnostic spans, averaged across GPU medians: two list-sort spans total about 0.674 ms/step before and 0.210 after; state packing about 0.335 before and 0.088 after. The persistent CUB path changes dispatch and workspace handling; this is not an isolated measurement of cudaMalloc alone. Gather and transfer work remains.',
        'The boundary-aware SF three-GPU map improves observed active max/mean from 1.124 to 1.004 and ghosts from 350 to 104, versus paper-balanced. Yet its measured runtime gain is only 1.3%, with a timing-repeat interval crossing zero. Improving allocation statistics does not guarantee a substantial critical-path reduction.',
        'On two GPUs, the SF measured-cost choice gains 8.3% against the baseline selected using training data. The community map is faster in evaluation than that training-selected baseline, leaving only about 1.8% difference versus community. Berkeley shows no consistent selector gain. Five timing samples are not independent traffic-demand tests.',
        'Original SF single GPU: 33.700 ms; best observed two-GPU map: 130.578 ms; best three-GPU map: 152.588 ms. Multi-GPU acceleration over the original single GPU remains unproven.',
        'Do not train a forecast model yet. Preserve the faster selectable backend, investigate remaining list/coordination costs, and require a held-out net policy opportunity. The charged oracle trials are expensive, but they are not a lower bound on a future cheap predictor.' ])
    page(pdf,'Nsight interpretation: optimize time, not stall percentages',[
        'Matched SF packing samples: median duration falls from 134.912 to 27.024 microseconds, while long-scoreboard stalls rise from 22.750% to 46.805%. The kernel is much faster despite the higher stall fraction; the denominator and instruction mix changed.',
        'Matched gather samples: long-scoreboard stalls fall from 90.935% to 10.750%, while duration changes only from 131.312 to 126.752 microseconds. A large reduction in this percentage does not imply a proportional runtime benefit. The experiment does not identify every replacement stall category.',
        'Motion samples remain similar: about 22-23 microseconds, 12% achieved occupancy, 3.6-3.7% SM throughput and 41-42% long-scoreboard stalls. Junction arbitration samples are about 3.5-3.9 microseconds. These are selected kernel replay samples, not utilization averages for the complete simulation.',
        'Per-GPU diagnostic tables retain packing, transfer, list maintenance and host wait intervals. The main measured implementation opportunities were local memory/list maintenance and dispatch overhead, not evidence that raw NVLink bandwidth was saturated. Sorting was timed with events; it was not included in the selected Nsight kernel filter.' ])
    for name,fig in figures:pdf.savefig(fig)
    for case in cases:
        d=timing[timing.case==case][['backend','policy','ranks','mode','median_ms','min_ms','max_ms']]
        table(pdf,case+' — independent timing results',d,'All values in ms / 60 steps. Original single GPU is backend=dense; legacy is the frozen local backend.')
        table(pdf,case+' — measured final allocations',allocation[(allocation.case==case)&(allocation.ranks>1)][['policy','ranks','active_max_mean','ghosts','processed_records','reserved_capacity']].rename(columns={'active_max_mean':'active max/mean','processed_records':'processed','reserved_capacity':'capacity'}),'Same physical traffic under every map; averages over five independent timing repetitions. Capacity is allocated headroom, not active work.')
    table(pdf,'Measured-cost selector: independent evaluation',selector[['case','ranks','selected','baseline_selected_from_train','optimistic_gain_pct','repeat_bootstrap_low_pct','repeat_bootstrap_high_pct']].rename(columns={'baseline_selected_from_train':'baseline','optimistic_gain_pct':'gain %','repeat_bootstrap_low_pct':'low %','repeat_bootstrap_high_pct':'high %'}),'Bootstrap interval covers five timing repetitions, not independent demand scenarios or generalization.')
    table(pdf,'Selector overhead charged',selector[['case','ranks','baseline_cold_ms','calibration_process_ms','all_map_construction_ms','charged_selector_ms']],'Cold costs include setup. Construction was measured on Jetson; calibration process costs were measured on the A100 host. Report separately; no deployment-host construction claim.')
    for mode in sorted({0,chosen}):
        d=gpu_rank[(gpu_rank.case=='sf_98304')&(gpu_rank['mode']==mode)]
        cols=['rank','pack','count_d2h','peer_copies','keys','sort','gather','junction_refresh','unpack']
        table(pdf,f'SF per-GPU exchange spans — mode {mode}',d[[c for c in cols if c in d]],'Mean ms/step, summed across applicable exchange phases; GPU-event spans may include submission gaps. Not additive with host waiting, and not an unprofiled critical path.')
    if len(ns):
        labels_metric={
            'gpu__time_duration.sum':'Kernel duration (microseconds)',
            'sm__throughput.avg.pct_of_peak_sustained_elapsed':'SM throughput (% peak)',
            'sm__warps_active.avg.pct_of_peak_sustained_active':'Achieved occupancy (%)',
            'smsp__warps_eligible.avg.per_cycle_active':'Eligible warps / scheduler cycle',
            'smsp__thread_inst_executed_per_inst_executed.ratio':'Active threads / instruction',
            'smsp__issue_active.avg.pct_of_peak_sustained_active':'Issue-active cycles (%)',
            'smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct':'Long-scoreboard stalls (%)',
            'l1tex__throughput.avg.pct_of_peak_sustained_elapsed':'L1/TEX throughput (% peak)',
            'lts__throughput.avg.pct_of_peak_sustained_elapsed':'L2 throughput (% peak)',
            'launch__registers_per_thread':'Registers / thread'}
        for family in ns.family.unique():
            values=ns[(ns.family==family)&ns['Metric Name'].isin(labels_metric)].copy()
            duration=values['Metric Name']=='gpu__time_duration.sum'
            factors={'nsecond':.001,'ns':.001,'usecond':1.,'us':1.,'msecond':1000.,'ms':1000.,'second':1e6,'s':1e6}
            for idx in values[duration].index:
                unit=values.at[idx,'Metric Unit']
                if unit not in factors:raise ValueError('Unknown duration unit '+unit)
                values.at[idx,'value']*=factors[unit]
            values['Metric Name']=values['Metric Name'].map(lambda x:labels_metric.get(x,x))
            d=values.pivot(index='Metric Name',columns='mode',values='value').reset_index()
            d.columns=['Metric']+[f'Mode {c}' for c in d.columns[1:]]
            table(pdf,'Nsight Compute — '+family,d,'Median across sampled invocations/devices. Duration converted to microseconds. Raw CSV preserves device IDs and units. Matched one-step initial state; kernel replay is not concurrent wall timing.')
    else:page(pdf,'Hardware counters',['No usable new Nsight counter samples were returned. See profile logs for availability/failure; older backend counters are not substituted.'])
    page(pdf,'Decision gate and reproducibility',[
        'Do not proceed to forecasting based solely on this pilot. The preregistered gate requires >=10% net complete-run improvement on held-out demand patterns, uncertainty excluding zero, and exact correctness. These two frozen workloads with independent timing repetitions do not establish that gate.',
        'Inspect framework improvements separately from allocator improvements. Compare original single GPU, local single GPU, and all policies on identical execution code. A faster multi-GPU backend is not automatically faster than the original single GPU.',
        'Reproduction: research/multigpu/exchange_study/PROTOCOL.md and source; artifacts/exchange-study-20261029 contains frozen inputs, hashes, local checks, cloud archive and billing. All cloud outputs are SHA-256 checked before this report is generated.',
        'The calibration charge includes trial-process wall time and all candidate-map construction. This is a measured-cost candidate within a finite policy set, not a guaranteed optimal schedule. Reuse can amortize costs, but must be reported explicitly rather than omitted.' ])
    bills=json.loads((ART/'billing-final.json').read_text())
    study_names={'lpsim-a100-partition-study','lpsim-local-scaling','lpsim-exchange-study'}
    cumulative=sum(float(r['cost']) for r in bills if r.get('description') in study_names)
    followup=sum(float(r['cost']) for r in bills if r.get('description')=='lpsim-exchange-study')
    page(pdf,'Hardware, spending and stopped-state receipt',[
        'Physical allocation: three NVIDIA A100-SXM4-80GB GPUs on one host. CUDA toolkit 12.6.3, sm_80 build; driver 610.57.04. UUIDs, NVLink status and clock/power telemetry are retained. The topology-matrix command failed, so exact switch wiring is not asserted.',
        f'Posted cost for this follow-up, including the failed collector attempt: ${followup:.4f}. Cumulative study cost: ${cumulative:.4f} of the authorized $10. Both apps are stopped with zero tasks; no live service is required to read this report.',
        'The first attempt passed its initial physical check, then stopped on a Python metrics-collector field collision. The wrapper was fixed; failed-attempt evidence and cost are preserved. Its timings are excluded.',
        'The report bundle contains the PDF, figures and CSVs. The replay bundle additionally includes frozen model/snapshot/map inputs, the exact physics source closure, experiment sources and the returned cloud archive. Frozen maps can be replayed without OSM downloads; regenerating the original geographic/community maps requires the separately preserved network geometry.' ])
for _,fig in figures:plt.close(fig)
receipt=dict(passed=True,chosen_mode=chosen,test_runs=len(tests),exact_reference_steps=valid_steps,
             nsight_metric_rows=len(nsight),heldout_demand=False,forecasting_gate_passed=False,
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(OUT/'findings.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2));print(timing.to_string(index=False));print(selector.to_string(index=False))
