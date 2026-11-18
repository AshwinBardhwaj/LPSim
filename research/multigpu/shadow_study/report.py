from pathlib import Path
import json,textwrap
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'reports/shadow-study-20261104';r=pd.read_csv(out/'window-comparison.csv');f=json.loads((out/'findings.json').read_text());plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False});figs=[]
def save(fig,name):
 fig.tight_layout();fig.savefig(out/(name+'.png'),dpi=160,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');figs.append(fig)
fig,axes=plt.subplots(2,2,figsize=(12,7))
for ax,(case,v) in zip(axes.flat,r.groupby('case')):
 for policy,color in [('reactive','#167c91'),('forecast','#e89439')]:
  z=v[v.policy==policy];ax.plot(z.decision_s.to_numpy(),z.raw_gain_pct.to_numpy(),'--o',color=color,label=policy+' · replay only');ax.plot(z.decision_s.to_numpy(),z.pre_migration_gain_pct.to_numpy(),'-s',color=color,label=policy+' · overhead charged')
 ax.axhline(0,color='k',lw=.7);ax.set_title(case.replace('test_','').replace('_',' '));ax.set_xlabel('Decision time (s)');ax.set_ylabel('Gain vs fixed (%) ↑');ax.grid(alpha=.15)
axes[0,0].legend(fontsize=8);fig.suptitle('Shadow proposals: replay benefit versus adaptation overhead',fontsize=16,y=1.03);save(fig,'01-window-gains')
fig,axes=plt.subplots(1,2,figsize=(12,5));labels=[];react=[];forecast=[]
for (case,t),v in r.groupby(['case','decision_s']):
 labels.append(case.replace('test_','').replace('_16384','')+' @ '+str(t));v=v.set_index('policy');react.append(v.loc['reactive','migration_budget_ms']);forecast.append(v.loc['forecast','migration_budget_ms'])
y=np.arange(len(labels));axes[0].barh(y-.18,react,.35,label='Reactive',color='#167c91');axes[0].barh(y+.18,forecast,.35,label='Forecast',color='#e89439');axes[0].set_yticks(y,labels);axes[0].axvline(0,color='k',lw=.8);axes[0].set_xlabel('Remaining budget for a live switch (ms)');axes[0].legend()
for policy,color in [('reactive','#167c91'),('forecast','#e89439')]:
 v=r[r.policy==policy];axes[1].scatter(v.changed_vehicle_authorities,v.migration_budget_ms,label=policy,color=color)
axes[1].axhline(0,color='k',lw=.8);axes[1].set_xlabel('Vehicles whose authority would change');axes[1].set_ylabel('Remaining switch budget (ms)');axes[1].set_title('Transfers counted; migration time not measured');save(fig,'02-migration-budget')
q=pd.read_csv(out/'forecast-quality.csv');fig,ax=plt.subplots(figsize=(9,4))
x=np.arange(12)
for i,(policy,color) in enumerate([('reactive','#167c91'),('forecast','#e89439')]):
 v=q[q.policy==policy].sort_values(['case','decision_s']);ax.bar(x+(i-.5)*.36,100*v.wape,.36,label=policy,color=color)
ax.set_xticks(x,labels,rotation=40,ha='right');ax.set_ylabel('Road-residency forecast WAPE (%) ↓');ax.legend();ax.set_title('Does the forecast improve on holding current occupancy?');save(fig,'03-forecast-error')
# Post-hoc startup-amortization sensitivity; does not replace the frozen gate.
r['warm_budget_ms']=r.migration_budget_ms+r.shared_preparation_ms
r['warm_gain_pct']=100*r.warm_budget_ms/r.fixed_step_ms
r.to_csv(out/'startup-sensitivity.csv',index=False)
warm={p:{'mean_gain_pct':float(v.warm_gain_pct.mean()),'positive_windows':int((v.warm_budget_ms>0).sum())} for p,v in r.groupby('policy')}
(out/'startup-sensitivity.json').write_text(json.dumps(warm,indent=2))
fig,ax=plt.subplots(figsize=(10,5))
components=[('replay_step_ms','Replay'),('construction_ms','Proposal'),('observation_ms','Observation'),('diagnostic_excess_ms','Diagnostic excess'),('shared_preparation_ms','Startup share')]
med=r.groupby('policy').mean(numeric_only=True);bottom=np.zeros(2)
for col,label in components:
 vals=med.loc[['reactive','forecast'],col].to_numpy();ax.bar(['Reactive','Forecast'],vals,bottom=bottom,label=label);bottom+=vals
ax.axhline(r.fixed_step_ms.mean(),color='black',ls='--',label='Fixed replay mean')
ax.set_ylabel('Mean wall time per 60 simulated seconds (ms)');ax.set_title('Startup dominates the charged prototype overhead');ax.legend(loc='upper left',bbox_to_anchor=(1,1));save(fig,'04-cost-breakdown')
with PdfPages(out/'shadow-study-report.pdf') as pdf:
 def page(title,paragraphs):
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.07,.92,title,size=21,weight='bold',color='#17364d');y=.85
  for p in paragraphs:
   lines=textwrap.wrap(p,115);fig.text(.07,y,'\n'.join(lines),size=11.5,va='top',linespacing=1.3);y-=len(lines)*.027+.025
  fig.text(.07,.03,'LPSim • shadow allocation pilot • three A100 GPUs • November 4, 2026',size=9,color='#64748b');pdf.savefig(fig);plt.close(fig)
 page('Can traffic feedback improve GPU ownership?',[
  'Fixed boundary-aware, reactive, and forecast-generated maps replay identical Berkeley snapshots. Twelve decision windows cover distributed, concentrated, mixed and bursty demand with 16,384 requests. Five shuffled timing repetitions replay 60 simulated seconds at each decision. The running reference never changes its ownership.',
  f"Reactive: {f['reactive']['raw']['mean_pct']:+.2f}% average replay-only gain; {f['reactive']['pre_migration']['mean_pct']:+.2f}% after measured construction and observation charges, before migration. Positive remaining switch budget in {f['reactive']['positive_budget_windows']}/12 windows.",
  f"Forecast: {f['forecast']['raw']['mean_pct']:+.2f}% replay-only gain; {f['forecast']['pre_migration']['mean_pct']:+.2f}% after pre-migration overhead. Positive switch budget in {f['forecast']['positive_budget_windows']}/12 windows. Forecast versus reactive after those costs: {f['forecast_vs_reactive']['mean_pct']:+.2f}%.",
  'A positive remaining budget means a real migration implementation would need to fit inside that amount. It is not proof of a net dynamic speedup. A negative budget rules out that proposal at the measured 60 s horizon under these prototype costs, even if migration were free.',
  f"Correctness: {f['checks']['reference_steps']} every-step physical-device reference checks; every timed and diagnostic final state matches the uninterrupted reference at its target simulation time. No future snapshots are read by the candidate generator."
 ])
 page('What the overhead result does—and does not—show',[
  'Mean replay gains are 9.38% for reactive maps (case-bootstrap 95% interval 1.46% to 20.66%) and 3.86% for forecast maps (0.14% to 8.28%). There is an allocation opportunity in these snapshots, but replay gains alone do not establish a deployable dynamic allocator.',
  'Median charges per decision: 369.42 ms topology preparation amortized over three decisions, 34.40 ms observation, 3.07 ms diagnostic excess, and 47.65 ms reactive / 50.58 ms forecast proposal construction. Fixed replay median is 553.58 ms for all 240 steps, not for one timestep. Startup is the dominant charge.',
  f"Post-hoc warm-cache sensitivity excludes only topology startup: reactive mean gain {warm['reactive']['mean_gain_pct']:+.2f}% with positive remaining migration budget in {warm['reactive']['positive_windows']}/12 windows; forecast {warm['forecast']['mean_gain_pct']:+.2f}% with {warm['forecast']['positive_windows']}/12 positive windows. Observation, proposal and diagnostic charges remain. This is not the preregistered gate and still excludes actual migration.",
  'Recommended next experiment: retain static topology across decisions, replace correctness-checkpoint monitoring with compact counters, and measure a reactive policy at multiple retention horizons. Only implement live switching after repeatable savings cover these recurring costs plus measured migration. Improve the forecast only if it can beat this reactive baseline.',
  'No new Modal runs were needed for this sensitivity analysis. Full startup charges remain in the primary results; no evidence is discarded to make the policy look faster.'
 ])
 for fig in figs:pdf.savefig(fig,bbox_inches='tight')
 page('Mechanism and accounting',[
  'Reactive weights use current owned records, including pending/completed records, and preceding rank-local elapsed GPU cost per visible record. This distributes rank averages across junctions; it is not direct per-junction cost measurement and can misattribute fixed topology costs. Cut weights represent current ghost residency, reservation messages and recent endpoint churn.',
  'Forecast weights use the preceding 30 s of membership change and known upcoming departures to estimate mean road occupancy over the next 60 s. Estimates are storage-bounded. Both candidates use weighted METIS and GPU-label matching that minimizes vehicle ownership changes. This is a simple causal forecast, not a learned policy; it does not explicitly model overdue pending admissions or propagate future vehicles along routes.',
  'Timing includes the complete simulation step: motion, both exchanges, junction arbitration, reservations and spawn. Candidate charges include CPU map/forecast construction, the preceding/current observation capture cost, positive measured diagnostic timing excess, and one topology-load charge per three-decision run. Checkpoint capture includes validation and disk output; these are conservative prototype charges, not a proven minimum monitoring cost.',
  'A live switch is not implemented. Changed vehicle IDs and 432-byte record payload give only part of its cost: ghost regeneration, index maintenance, junction/reservation transfer and synchronization remain. Cold replay process/setup timings are saved separately and are not relabeled migration time.',
  'These four demand cases were inspected in the earlier study. This is exploratory mechanism evaluation, not fresh held-out confirmation. Confidence intervals resample four demand cases, keeping their windows grouped. The preregistered pilot gate is at least 5% gain with a positive lower bound before migration; passing would justify measuring migration, not claiming deployment speedup.',
  f"Road-residency WAPE: current-state rule {100*f['reactive']['forecast_wape']:.1f}%; forecast {100*f['forecast']['forecast_wape']:.1f}%. No Nsight hardware counters are claimed; diagnostic timings use CUDA events."
 ])
text='# Shadow-mode allocation pilot\n\nSee [PDF report](shadow-study-report.pdf), window-comparison.csv and findings.json.\n\n'
for policy in ['reactive','forecast']:
 a=f[policy];text+=f"- {policy}: replay-only {a['raw']['mean_pct']:+.2f}%; pre-migration {a['pre_migration']['mean_pct']:+.2f}%; positive switch budget {a['positive_budget_windows']}/12.\n"
text+='\nNo live ownership changes occurred. Migration budgets are break-even amounts, not measured switching latencies. Forecast generation uses no future simulated states.\n';(out/'README.md').write_text(text);print(out/'shadow-study-report.pdf')
