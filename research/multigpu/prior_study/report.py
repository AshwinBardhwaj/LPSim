from pathlib import Path
import json,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'reports/prior-study-20261103';f=json.loads((out/'findings.json').read_text());s=pd.read_csv(out/'policy-summary.csv');alltiming=pd.read_csv(out/'timings.csv');sel=pd.read_csv(out/'heldout-selection.csv');pop=pd.read_csv(out/'populations.csv');win=pd.read_csv(out/'windows.csv')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
colors={'geography':'#64748b','paper_balanced':'#167c91','boundary_aware':'#e89439'};figures=[]
def save(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');figures.append(fig)
# Show the actual frozen demand plans rather than a decorative network image.
import struct
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
inputs=ROOT/'artifacts/prior-study-20261103/inputs';geo=json.loads((inputs/'geometry.json').read_text());roads=geo['roads'];segments=np.array([[r['start'],r['end']] for r in roads]);CAR=struct.Struct('<102i5f');plans=[]
for pattern in ['distributed','concentrated','mixed','bursty']:
 raw=(inputs/f'test_{pattern}_16384/snapshot.bin').read_bytes();cars=[CAR.unpack_from(raw,8+i*CAR.size) for i in range(16384)];visits=np.zeros(len(roads))
 for c in cars:np.add.at(visits,list(c[6:6+c[2]]),1)
 plans.append((pattern,visits,[c[102] for c in cars]))
fig,axes=plt.subplots(2,4,figsize=(14,7),gridspec_kw={'height_ratios':[3,1]});norm=LogNorm(1,max(v.max() for _,v,_ in plans))
for j,(pattern,visits,departures) in enumerate(plans):
 ax=axes[0,j];ax.add_collection(LineCollection(segments,colors='#dbe2e5',linewidths=.35));lc=LineCollection(segments[visits>0],array=visits[visits>0],cmap='YlOrRd',norm=norm,linewidths=.7);ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal');ax.set_axis_off();ax.set_title(pattern.capitalize());axes[1,j].hist(departures,bins=np.arange(0,601,30),color='#167c91');axes[1,j].set_xlabel('Departure time (s)');axes[1,j].set_ylabel('Trips / 30 s')
fig.suptitle('Frozen Berkeley demand plans • 16,384 requested trips\nRoad color: planned route traversals (log scale); maps © OpenStreetMap contributors',fontsize=14,y=1.03);save(fig,'00-demand-and-network')
fig,axes=plt.subplots(2,4,figsize=(14,6),sharex=True)
for i,n in enumerate([4096,16384]):
 for j,pattern in enumerate(['distributed','concentrated','mixed','bursty']):
  ax=axes[i,j];p=pop[(pop.split=='test')&(pop.n==n)&(pop.pattern==pattern)&(pop.policy=='geography')].groupby('window').sum(numeric_only=True);x=(p.index+1)*30
  for col,c in [('pending','#a4acb7'),('stopped','#d64a48'),('moving','#167c91'),('connector','#dda22d'),('completed','#5352a4')]:ax.plot(np.asarray(x),p[col].to_numpy(),label=col,color=c,lw=1.8)
  ax.set_title(f'{pattern.capitalize()} · {n:,} requested');ax.set_xlabel('Simulation time (s)');ax.set_ylabel('Vehicles');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.suptitle('Actual traffic populations • held-out runs',fontsize=16,y=1.02);save(fig,'01-traffic-populations')
fig,axes=plt.subplots(2,4,figsize=(14,6))
for i,n in enumerate([4096,16384]):
 for j,pattern in enumerate(['distributed','concentrated','mixed','bursty']):
  ax=axes[i,j];v=s[(s.split=='test')&(s.n==n)&(s.pattern==pattern)].set_index('policy');names=list(colors);bars=ax.bar(range(3),[v.loc[k,'complete_ms']/1000 for k in names],color=[colors[k] for k in names]);vv=alltiming[(alltiming.split=='test')&(alltiming.n==n)&(alltiming.pattern==pattern)];med=np.array([v.loc[k,'complete_ms']/1000 for k in names]);lo=np.array([vv[vv.policy==k].complete_ms.min()/1000 for k in names]);hi=np.array([vv[vv.policy==k].complete_ms.max()/1000 for k in names]);ax.errorbar(range(3),med,yerr=[med-lo,hi-med],fmt='none',color='#333333',capsize=3);ax.set_xticks(range(3),['Geo','Paper','Boundary']);ax.set_title(f'{pattern} · {n:,}');ax.set_ylabel('Complete cost (s)');[ax.text(k,hi[k]+.02*max(hi),f'{med[k]:.2f}',ha='center',fontsize=8) for k in range(3)];ax.set_ylim(0,max(hi)*1.2)
fig.suptitle('600 s simulation • median of three independent timing runs\nIncludes process wall time and selected partition construction; whiskers: repeat range',fontsize=14,y=1.05);save(fig,'02-policy-comparison')
fig,axes=plt.subplots(1,2,figsize=(12,4.8));labels=['Fixed calibration winner','Measured-cost selector','Prior, schedule supplied','Prior, all costs charged','Hindsight best (upper reference)'];cols=['fixed_ms','measured_ms','prior_supplied_ms','prior_ms','hindsight_ms'];vals=[100*np.mean(sel[c]/sel.fixed_ms) for c in cols]
bars=axes[0].barh(labels,vals,color=['#64748b','#167c91','#79b1bb','#e89439','#a9b0ba']);axes[0].invert_yaxis();axes[0].axvline(100,color='k',lw=.7);axes[0].bar_label(bars,fmt='%.1f%%');axes[0].set_xlim(0,max(vals)*1.2);axes[0].set_xlabel('Held-out mean cost / fixed baseline (%) ↓')
stages=['motion_ms','exchange0_ms','junction_ms','reservation_ms','spawn_ms','exchange1_ms'];vv=win[(win.split=='test')&(win.policy==f['fixed_policy'])].groupby(['case','repeat'])[stages].sum().mean();axes[1].barh(stages,vv/1000,color='#167c91');axes[1].set_xlabel('Mean stage time per 600 s run (s)');axes[1].set_title('Where step time goes');save(fig,'03-selection-and-costs')
fig,axes=plt.subplots(1,3,figsize=(13,4.2))
for ax,name,target,title in zip(axes,['regional-compute','boundary-volume','window-runtime'],['regional_gpu_ms','wire_bytes','total_ms'],['Regional GPU work (ms/step)','Boundary volume (bytes/30 s)','Window runtime (ms/30 s)']):
 p=pd.read_csv(out/(name+'-predictions.csv'));p=p[p.split=='test'];ax.scatter(p[target],p.prediction,s=9,alpha=.35,color='#167c91');lim=max(p[target].max(),p.prediction.max());ax.plot([0,lim],[0,lim],color='#d64a48',lw=1);ax.set_xlabel('Measured');ax.set_ylabel('Predicted');ax.set_title(title)
fig.suptitle('Forecasts use only topology, routes and departure schedules\nUnseen demand seed; same Berkeley network',fontsize=14,y=1.05);save(fig,'04-heldout-predictions')
fig,ax=plt.subplots(figsize=(12,5));winner=pd.read_csv(out/'window-winners.csv');winner=winner[winner.case.str.startswith('test')];cases=sorted(winner.case.unique());codes={p:i for i,p in enumerate(colors)};arr=np.array([[codes[winner[(winner.case==c)&(winner.window==k)].policy.iloc[0]] for k in range(20)] for c in cases]);from matplotlib.colors import ListedColormap
ax.imshow(arr,aspect='auto',cmap=ListedColormap(list(colors.values())),vmin=0,vmax=2,extent=[0,600,len(cases)-.5,-.5]);ax.set_yticks(range(len(cases)),[c.replace('test_','').replace('_',' ') for c in cases]);ax.set_xlabel('Simulation time (s)');ax.set_title('Fastest policy in each 30 s window • exploratory median timings');from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c,label=p) for p,c in colors.items()],loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3);save(fig,'05-policy-persistence')
fig,axes=plt.subplots(3,2,figsize=(12,7),sharex=True)
rg=pd.read_csv(out/'regional-gpu.csv')
for i,policy in enumerate(colors):
 for j in range(2):
  df=pop if j==0 else rg;v=df[(df.case=='test_concentrated_16384')&(df.policy==policy)].copy()
  if j==0:v['value']=v.stopped+v.moving+v.connector
  else:v['value']=v.regional_gpu_ms
  matrix=v.pivot(index='rank',columns='window',values='value');im=axes[i,j].imshow(matrix,aspect='auto',extent=[0,600,2.5,-.5],cmap='viridis');axes[i,j].set_yticks([0,1,2],['GPU 0','GPU 1','GPU 2']);axes[i,j].set_title(policy+(' · active vehicles' if j==0 else ' · local GPU elapsed ms/step'));fig.colorbar(im,ax=axes[i,j],shrink=.75)
for ax in axes[-1]:ax.set_xlabel('Simulation time (s)')
fig.suptitle('Allocation over time · concentrated demand · 16,384 requested',fontsize=15,y=1.02);save(fig,'06-gpu-allocation')
with PdfPages(out/'prior-study-report.pdf') as pdf:
 def textpage(title,paras):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.92,title,size=22,weight='bold',color='#17364d');y=.85
  for paragraph in paras:
   lines=textwrap.wrap(paragraph,112);fig.text(.07,y,'\n'.join(lines),size=11.5,va='top',linespacing=1.3);y-=.027*len(lines)+.025
  fig.text(.07,.035,'LPSim · three A100 GPUs · Berkeley · frozen protocol and raw tables included',size=9,color='#64748b');pdf.savefig(fig);plt.close(fig)
 b=f['selection'];textpage('Traffic priors: held-out allocation study',[
  'Experiments 3–5: longer varied traffic, measured-cost partition selection, and a topology/demand-only cost predictor. The backend remains full-rebuild exchange mode 3; no memory-layout or physics optimization is mixed into policy comparisons.',
  f"Fixed calibration winner: {f['fixed_policy']}. Measured-cost selection: {b['measured']['mean_gain_pct']:+.2f}% mean held-out cost reduction; 95% case-bootstrap interval {b['measured']['ci95_pct'][0]:+.2f}% to {b['measured']['ci95_pct'][1]:+.2f}%.",
  f"Prior, including all map/feature costs: {b['prior']['mean_gain_pct']:+.2f}% held-out reduction. Supplied-schedule diagnostic: {b['prior_supplied']['mean_gain_pct']:+.2f}%. Hindsight best candidate: {b['hindsight']['mean_gain_pct']:+.2f}%. Positive means faster; negative means slower.",
  f"Preregistered gate: at least 5% net improvement and a positive lower confidence bound. Measured selector passed: {b['measured']['passes_gate']}; prior passed: {b['prior']['passes_gate']}.",
  'Ownership is static within each run. Ordinary vehicle migration is included; repartition migration is zero because no repartition occurs. These results do not establish benefits for dynamic ownership changes or another network.'
 ])
 for fig in figures:pdf.savefig(fig,bbox_inches='tight')
 textpage('What the traffic conditions explain',[
  'The held-out 16,384-request workloads reached 13,097–15,063 active vehicles; the 4,096-request workloads reached 3,729–3,944. These are substantial active populations, not an increase confined to pending demand.',
  'Against geography, boundary-aware complete cost improved 24.52% for distributed, 9.77% for concentrated, 27.28% for mixed and 29.35% for bursty high-demand runs. At 4,096 requests the same comparison ranged from 4.32% slower to 1.79% faster. Traffic intensity changes the value of partitioning.',
  'Paper-balanced was faster for the held-out distributed and bursty cases at both request counts. Boundary-aware won both mixed cases and the 16,384-request concentrated case. Geography won the 4,096-request concentrated case. These are observed medians, not proof that each difference is significant.',
  'A key transfer failure: paper-balanced won the calibration mixed 16,384 case but cost 9.424 s on its test seed, versus 6.883 s for boundary-aware. Its mean active imbalance changed from 1.31 to 1.46; ghost records from 299 to 336. Different sampled routes generate different maps. A pattern label alone did not reliably transfer the winning decision.',
  f"Held-out prediction WAPE: regional GPU elapsed work {100*f['models']['regional_compute']['wape']:.1f}%; boundary volume {100*f['models']['boundary_volume']['wape']:.1f}%; 30 s window runtime {100*f['models']['window_runtime']['wape']:.1f}%. Runtime error exceeds the small average selection opportunity. Regional cost-change MAE was {f['persistence_delta_mae_ms']:.3f} ms, versus {f['persistence_constant_cost_mae_ms']:.3f} ms for predicting no change.",
  'Recommendation: retain boundary-aware as the fixed calibration-selected reference for this workload family. Before expanding a predictor, test independent seeds/maps where candidate policies have larger repeatable differences. Do not infer dynamic repartitioning gains from a hindsight window winner; migration and switching remain untested.'
 ])
 textpage('Method, accounting and limits',[
  '16 independent demand cases: four patterns × two requested populations (4,096 and 16,384) × calibration/test seeds. Each case runs 600 s at dt=0.25 s, fixed 72 s controller. Three candidate policies share identical physics and buffers. Three shuffled timing repeats; diagnostics excluded from timing aggregates.',
  'All vehicles start pending. Demand is synthetic and modestly arterial-biased, not a calibrated OD estimate. Concentrated demand also departs earlier; pattern comparisons therefore describe combined scenarios, not isolated causal effects of spatial concentration.',
  'Per-rank pools reserve N vehicle records for safe longer runs. Populations and physical/ghost checks sampled every 30 s. Regional CUDA-event samples cover four steps per 30 s. They are GPU elapsed times, not Nsight long-scoreboard, occupancy or SM-throughput counters.',
  'Measured selector uses calibration winners by pattern/population. Prior uses fixed ridge regression (alpha=10) with no test-set tuning. Its route-residence proxy assumes free-flow travel plus 10 s approach dwell, so queues can invalidate its population forecast.',
  f"One-time calibration timing cost: {f['calibration_timing_total_ms']/1000:.2f} s across all repeated calibration jobs, before diagnostics. This is reported separately, not hidden in claimed amortized per-run benefits. Startup, selected map cost and selection are included; prior additionally pays all candidate-map/feature costs.",
  'Confidence intervals bootstrap eight held-out cases, not individual timesteps. Same-network unseen-seed validation is limited evidence of generalization. Winning-window colors can reflect measurement noise; they do not by themselves justify online switching.',
  'Supplementary same-map check (post-hoc): choose on repeat 0, evaluate on repeats 1–2 of identical demand/maps. Average supplied-schedule gain is 2.71%; charging all calibration runs and candidate construction makes a cold use 4.34 times the fixed baseline on average. This is distinct from transferring a policy across unseen demand seeds.'
 ])
text='''# Longer traffic and prior-selection study\n\nSee [PDF report](prior-study-report.pdf), policy-summary.csv, heldout-selection.csv, traffic-summary.csv, regional-gpu.csv, and findings.json.\n\n'''
text+=f"Fixed calibration winner: **{f['fixed_policy']}**.\n\n"
for label,m in f['selection'].items():text+=f"- {label}: {m['mean_gain_pct']:+.2f}% mean gain, 95% CI {m['ci95_pct'][0]:+.2f} to {m['ci95_pct'][1]:+.2f}%; gate passed: {m['passes_gate']}.\n"
text+='\nHigh-demand boundary-aware gains versus geography: 24.52% distributed, 9.77% concentrated, 27.28% mixed, 29.35% bursty. Actual peak active populations: 13,097–15,063. Lower-demand results are mixed.\n\nSupplementary same-map selector: +2.71% supplied-schedule gain; calibration charged cold cost exceeds baseline. See same-map-selector.csv.\n\nStatic pre-run selection only. No online repartitioning. CUDA-event measurements are not Nsight hardware counters. Raw forecasts and frozen model coefficients are included.\n'
(out/'README.md').write_text(text)
print(out/'prior-study-report.pdf')
