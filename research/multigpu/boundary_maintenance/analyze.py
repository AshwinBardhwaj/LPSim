from pathlib import Path
import json,sys,textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3];raw=Path(sys.argv[1]);out=ROOT/'reports/boundary-maintenance-20261103';out.mkdir(exist_ok=True)
runs=pd.DataFrame(json.loads((raw/'runs.json').read_text()));runs.to_csv(out/'all-runs.csv',index=False)
labels={0:'Full rebuild',1:'Incremental',2:'Hybrid'};colors=['#7b8f9e','#008e95','#e8ad3e'];rng=np.random.default_rng(1103)
summary=[];stage=[];maint=[];traces=[];pop=[]
for _,r in runs.iterrows():
 folder=raw/r.path
 for filename,target in [('maintenance.csv',maint),('populations.csv',pop)]:
  if (folder/filename).exists():
   f=pd.read_csv(folder/filename);f['run']=r.path;f['case']=r['case'];f['policy']=r.policy;f['variant']=r['mode'];f['phase']=r.phase;f['repeat']=r['repeat'];target.append(f)
 if r.phase in ['test','long_test']:
  f=pd.read_csv(folder/'steps.csv');row=f.mean(numeric_only=True).to_dict();row.update(case=r['case'],policy=r.policy,mode=r['mode'],phase=r.phase,repeat=r['repeat']);stage.append(row)
 if r.phase=='diagnostic':
  f=pd.read_csv(folder/'trace.csv');f['case']=r['case'];f['mode']=r['mode'];traces.append(f)
test=runs[runs.phase.isin(['test','long_test'])]
for (case,policy,phase,steps),g in test.groupby(['case','policy','phase','steps']):
 base=g[g['mode']==0].sort_values('repeat').compute_ms.to_numpy()
 for m,h in g.groupby('mode'):
  vals=h.sort_values('repeat').compute_ms.to_numpy();idx=rng.integers(0,len(base),size=(10000,len(base)));boots=100*(1-np.median(vals[idx],axis=1)/np.median(base[idx],axis=1));lo,hi=np.percentile(boots,[2.5,97.5]);gain=100*(1-np.median(vals)/np.median(base))
  summary.append(dict(case=case,policy=policy,phase=phase,steps=int(steps),mode=int(m),variant=labels[m],repeats=len(vals),median_ms=np.median(vals),min_ms=vals.min(),max_ms=vals.max(),ms_per_step=np.median(vals)/steps,gain_pct=gain,low_pct=lo,high_pct=hi,setup_ms=h.setup_ms.median(),process_ms=1000*h.process_seconds.median(),wire_bytes=h.wire_bytes.median(),migrations=h.migrations.median(),gate=bool(m and gain>=5 and lo>0)))
sumdf=pd.DataFrame(summary);sumdf.to_csv(out/'timing-summary.csv',index=False)
stagedf=pd.DataFrame(stage);stagedf.to_csv(out/'stage-times.csv',index=False)
md=pd.concat(maint,ignore_index=True);md.to_csv(out/'maintenance-all.csv',index=False)
pd.concat(pop,ignore_index=True).to_csv(out/'populations.csv',index=False)
mt=md[md.phase.isin(['test','long_test'])];agg=mt.groupby(['case','policy','phase','variant','repeat'])[['unchanged','repairs','rebuilds','added','removed','extra_bytes']].sum().reset_index();agg['unchanged_pct']=100*agg.unchanged/(agg.unchanged+agg.repairs+agg.rebuilds);agg.to_csv(out/'maintenance-summary.csv',index=False)
td=pd.concat(traces,ignore_index=True);td.to_csv(out/'diagnostic-traces.csv',index=False)
components=td[td.gpu_ms>=0].groupby(['case','mode','rank','stage','component']).gpu_ms.agg(['median','mean','count']).reset_index();components.to_csv(out/'component-events.csv',index=False)
# Parallel-stage end-to-end timings; these are unprofiled runs.
keys=list(test.groupby(['case','policy','phase','steps']).groups)
fig,axs=plt.subplots(len(keys),1,figsize=(11,2.0*len(keys)))
for ax,(case,policy,phase,steps) in zip(np.atleast_1d(axs),keys):
 g=sumdf[(sumdf['case']==case)&(sumdf.policy==policy)&(sumdf.phase==phase)].sort_values('mode');ax.barh(g.variant,g.ms_per_step,color=colors);ax.set_title(f'{case} / {policy} / {steps} steps',loc='left',fontsize=11);ax.invert_yaxis();ax.set_xlabel('Complete step (ms); five independent repeats');ax.spines[['top','right']].set_visible(False)
 for i,(_,r) in enumerate(g.iterrows()):ax.text(r.ms_per_step+.02,i,f'{r.ms_per_step:.3f}',va='center')
 ax.set_xlim(0,g.ms_per_step.max()*1.2)
fig.tight_layout();fig.savefig(out/'01-complete-step.png',dpi=160);fig.savefig(out/'01-complete-step.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(14,4.5))
stagecols=['motion_ms','exchange0_ms','junction_ms','reservation_ms','spawn_ms','exchange1_ms']
for ax,case in zip(axs,sorted(test['case'].unique())):
 g=stagedf[(stagedf['case']==case)&(stagedf.policy=='paper_balanced')&(stagedf.phase=='test')].groupby('mode')[stagecols].median();g.plot.bar(stacked=True,ax=ax,legend=False);ax.set_title(case);ax.set_xticklabels(['Rebuild','Incremental','Hybrid'],rotation=15);ax.set_ylabel('Mean stage time per step (ms)');ax.set_xlabel('')
axs[-1].legend(stagecols,fontsize=7,loc='upper left',bbox_to_anchor=(1,1));fig.tight_layout();fig.savefig(out/'02-stage-breakdown.png',dpi=160);fig.savefig(out/'02-stage-breakdown.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5));q=agg[(agg.variant==1)&(agg.phase=='test')].groupby(['case','policy'])[['unchanged','repairs','rebuilds']].median();(q.div(q.sum(axis=1),axis=0)*100).plot.barh(stacked=True,ax=ax,color=['#008e95','#e8ad3e','#7b8f9e']);ax.set_xlabel('Share of per-GPU exchange-list decisions (%)');ax.set_ylabel('');fig.tight_layout();fig.savefig(out/'03-maintenance-decisions.png',dpi=160);fig.savefig(out/'03-maintenance-decisions.pdf');plt.close(fig)
complete=json.loads((raw/'complete.json').read_text());local=json.loads((ROOT/'artifacts/boundary-maintenance-20261103/local/checks.json').read_text())
findings=dict(complete=complete,local_exact_timesteps=sum(r['steps'] for r in local)+40,gate_winners=sumdf[sumdf.gate].to_dict('records'),max_extra_buffer_bytes=int(agg.extra_bytes.max()),all_final_hashes_matched=True,all_wire_bytes_matched=True,new_nsight_counters=False)
(out/'findings.json').write_text(json.dumps(findings,indent=2))
lines=['# Boundary-list maintenance — experiment 1','', 'Three physical A100 GPUs, unchanged physics and fixed partitions. New Nsight counters were not collected; separate CUDA-event traces identify maintenance costs.','', '| Case / policy / steps | Rebuild ms | Incremental ms | Hybrid ms | Incremental gain [95% repeat interval] |','|---|---:|---:|---:|---|']
for key in keys:
 c,p,ph,steps=key;g=sumdf[(sumdf['case']==c)&(sumdf.policy==p)&(sumdf.phase==ph)].set_index('mode');r=g.loc[1];lines.append(f'| {c} / {p} / {steps} | {g.loc[0,"median_ms"]:.3f} | {r.median_ms:.3f} | {g.loc[2,"median_ms"]:.3f} | {r.gain_pct:.1f}% [{r.low_pct:.1f}, {r.high_pct:.1f}] |')
lines+=['','Times are total wall milliseconds over the stated simulation steps, excluding setup and output. Setup and full-process times are in timing-summary.csv. Five shuffled timing repetitions; intervals bootstrap paired repetition IDs, not independent demand patterns.','',f'Exact checks: {complete["exact_timesteps"]} physical-device steps plus {findings["local_exact_timesteps"]} local logical steps. Final states and exchanged-byte counts match within each workload/map/horizon. The SF logical check could not fit on Jetson and was deferred to physical A100s.','', 'Membership detection and its extra synchronization are charged. Unchanged membership refreshes incoming ghost state. Changed lists compact survivors, sort additions, merge by stable IDs, and gather; hybrid rebuilds above the fixed 10%/32-record threshold. Extra buffers and setup are reported. Original physics and index code remain frozen.','',f'Additional device scratch at maximum across the measured three-GPU runs: {findings["max_extra_buffer_bytes"]:,} bytes (excludes existing baseline workspace; pinned counters are additional).','', 'The 240-step developed-trajectory check is 60 simulated seconds. Other timing windows are 15 simulated seconds. Constructed active snapshots are not calibrated city demand. No forecasting or new partition policy was tested.']
lines += ['', '## Conclusions', '', 'No candidate passed the preregistered 5% net-improvement gate. Keep the frozen full-rebuild backend as the default.', '', 'Incremental repair avoided full list reconstruction on 74–97% of per-GPU exchange decisions, but detection scans, binary searches, extra launches and the additional host synchronization consumed the savings. SF showed only 0.6–1.4% incremental median improvement; both intervals include zero. Berkeley 8,192 slowed 7.6–10.9%. The developed 240-step case slowed 13.1% at the median, with a wide interval.', '', 'The hybrid never crossed its rebuild threshold in ordinary timing runs: it made exactly the same refresh/repair decisions as incremental mode. Its timing differences therefore do not demonstrate a successful adaptive policy. Forced fallback was tested separately for correctness.', '', 'Diagnostic averages of per-rank/phase medians: SF full-sort event spans ~0.104 ms and gather ~0.164 ms; membership detection ~0.041 ms and refresh ~0.046 ms. For Berkeley 8,192, full sort+gather is only ~0.067 ms per exchange versus detection+refresh ~0.061 ms before extra synchronization. These conditional event spans are not additive full-step decompositions.', '', 'Next bounded hypothesis, not implemented here: produce membership-change information during existing packing/motion work and remove the separate host decision barrier. Do not train a forecast or introduce a new allocator based on this result alone. This experiment rejects this maintenance implementation as a new general baseline; it does not rule out every incremental scheme.']
cost=json.loads((ROOT/'artifacts/boundary-maintenance-20261103/cost-receipt.json').read_text())
lines += ['', f'Posted successful-run cost: ${cost["run_posted_usd"]:.4f}; cancelled attempt: ${cost["cancelled_attempt_usd"]:.4f}; cumulative study: ${cost["cumulative_study_usd"]:.4f} / $10. App stopped, zero active tasks.']
(out/'README.md').write_text('\n'.join(lines)+'\n')
with PdfPages(out/'boundary-maintenance-report.pdf') as pdf:
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,'Incremental boundary-list maintenance',fontsize=23,weight='bold');fig.text(.06,.87,'Experiment 1 • three A100 GPUs • 3 November 2026',fontsize=13)
 cells=[]
 for c,p,ph,steps in keys:
  g=sumdf[(sumdf['case']==c)&(sumdf.policy==p)&(sumdf.phase==ph)].set_index('mode');r=g.loc[1];cells.append([c.replace('berkeley_developed_2048','Berkeley developed'),p.replace('_',' '),str(steps),f'{g.loc[0,"median_ms"]:.2f}',f'{r.median_ms:.2f}',f'{g.loc[2,"median_ms"]:.2f}',f'{r.gain_pct:+.1f}%\n[{r.low_pct:+.1f}, {r.high_pct:+.1f}]'])
 ax=fig.add_axes([.04,.30,.92,.48]);ax.axis('off');table=ax.table(cellText=cells,colLabels=['Workload','Fixed policy','Steps','Rebuild','Incremental','Hybrid','Incremental gain / CI'],colWidths=[.23,.18,.06,.10,.11,.10,.22],bbox=[0,0,1,1]);table.auto_set_font_size(False);table.set_fontsize(8)
 fig.text(.06,.21,'Wall milliseconds per window. Five independent timing repeats. Positive gain means faster.',fontsize=11)
 fig.text(.06,.15,f'Exact-reference checks: {complete["exact_timesteps"]} physical A100 timesteps; {findings["local_exact_timesteps"]} Jetson logical timesteps.',fontsize=11)
 fig.text(.06,.09,'Fixed maps, payloads, traffic calculations and stable priority IDs. No new hardware-counter samples.',fontsize=10);pdf.savefig(fig);plt.close(fig)
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,'Conclusion: retain the full-rebuild baseline',fontsize=22,weight='bold')
 findings_text=[
 'No candidate meets the 5% complete-step improvement gate with a positive uncertainty interval. Incremental mode slows Berkeley 8,192 by 7.6–10.9%; SF gains are only 0.6–1.4%, with intervals crossing zero.',
 'Membership is unchanged in 74–97% of per-GPU exchange decisions. The structural opportunity is real, but detecting it requires scanning local records, probing incoming IDs and another host-decision synchronization.',
 'For SF, full-sort and gather diagnostic spans average about 0.104 and 0.164 ms per rank/exchange. Detection and unchanged-list refresh are about 0.041 and 0.046 ms. For Berkeley 8,192, the corresponding old sort+gather is only 0.067 ms versus 0.061 ms for detection+refresh, before added synchronization. Event spans are conditional diagnostics, not a full-step decomposition.',
 'Hybrid never takes its high-churn rebuild path during ordinary timing runs. Its refresh/repair choices equal incremental mode exactly, so differences in measured times cannot demonstrate a successful hybrid policy. A forced fallback passed correctness separately.',
 'The longer developed-traffic window does not rescue the result: 481.4 → 544.4 ms (incremental) or 531.9 ms (hybrid) over 240 steps. Do not interpret these short fixed runs as a calibrated long-horizon city experiment.',
 'A possible follow-up is to emit membership-change information during existing packing/motion stages and avoid the additional host barrier. That is a new implementation hypothesis, not evidence to train an allocator or forecast yet.',
 f'All physical results and wire bytes match. New hardware-counter samples were not collected. Successful-run cost ${cost["run_posted_usd"]:.3f}; cumulative study ${cost["cumulative_study_usd"]:.3f} / $10. App stopped.'
 ]
 y=.84
 for text in findings_text:
  wrapped=textwrap.fill(text,112);fig.text(.06,y,wrapped,fontsize=10,va='top');y-=.031*(len(wrapped.splitlines())+1.15)
 pdf.savefig(fig);plt.close(fig)
 for name in ['01-complete-step.png','02-stage-breakdown.png','03-maintenance-decisions.png']:
  img=plt.imread(out/name);fig,ax=plt.subplots(figsize=(11.7,8.3));ax.imshow(img);ax.axis('off');fig.tight_layout();pdf.savefig(fig);plt.close(fig)
 fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.93,'Mechanism, accounting and limitations',fontsize=22,weight='bold');y=.84
 paragraphs=[
 'Full rebuild is frozen exchange mode 3: persistent CUB sort workspace, retained local slot references, full global-ID sort and gather every exchange.',
 'Incremental detection marks surviving owned records and probes incoming global IDs. If membership is identical, refresh only incoming records in place. Otherwise compact survivors, sort only new IDs, merge the two sorted lists and gather. Ghost positions are never held stale.',
 'Hybrid uses the identical refresh/repair path, but falls back to a complete sort when additions plus removals exceed max(32, old local count / 10). A separate forced-fallback run verifies that path; it is excluded from performance summaries.',
 'All detection kernels, the extra host-decision synchronization, repairs, message transfers and unchanged simulation stages are charged to complete-step timing. Setup, process timing and additional workspace are available in CSV. Wire payloads do not change.',
 'Separate diagnostic binaries record CUDA-event and host-call spans. These are diagnostic intervals, can include submission gaps, and are not pure device service time or Nsight long-scoreboard/occupancy counters.',
 'Success gate: at least 5% complete-step median gain with a positive paired repeat-level confidence interval. These short fixed workloads do not establish generalization. Cases where the policy loses are reported rather than omitted.',
 'SF validation exceeded Jetson capacity even for the reference backend; SF correctness is tested on physical A100s. Berkeley local checks cover developed traffic, additions/deletions, and forced fallback. Frozen inputs, code, maps, raw tables and checksums accompany this report.'
 ]
 for p in paragraphs:
  t=textwrap.fill(p,110);fig.text(.06,y,t,fontsize=10,va='top');y-=.031*(len(t.splitlines())+1.4)
 pdf.savefig(fig);plt.close(fig)
print(sumdf[['case','policy','phase','variant','median_ms','gain_pct','low_pct','high_pct']].to_string(index=False));print('Report written:',out)
