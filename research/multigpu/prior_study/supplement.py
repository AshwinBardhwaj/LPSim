"""Post-hoc same-map diagnostic; does not change the preregistered gate."""
from pathlib import Path
import pandas as pd,json
p=Path(__file__).resolve().parents[3]/'reports/prior-study-20261103';t=pd.read_csv(p/'timings.csv');rows=[]
for case,v in t[t.split=='test'].groupby('case'):
 calibration=v[v.repeat==0].set_index('policy');chosen=calibration.complete_ms.idxmin();evaluation=v[v.repeat.isin([1,2])].groupby('policy').median(numeric_only=True);fixed=evaluation.loc['boundary_aware','complete_ms'];cold_cost=calibration.wall_ms.sum()+calibration.map_ms.sum()+evaluation.loc[chosen,'wall_ms']
 rows.append(dict(case=case,chosen=chosen,fixed_policy='boundary_aware',fixed_evaluation_ms=fixed,selected_evaluation_ms=evaluation.loc[chosen,'complete_ms'],supplied_gain_pct=100*(1-evaluation.loc[chosen,'complete_ms']/fixed),all_calibration_and_construction_ms=cold_cost,charged_gain_pct=100*(1-cold_cost/fixed)))
x=pd.DataFrame(rows);x.to_csv(p/'same-map-selector.csv',index=False);f=json.loads((p/'findings.json').read_text());f['supplementary_same_map_selector']=dict(status='post-hoc diagnostic; not the preregistered held-out-demand gate',calibration_repeat=0,evaluation_repeats=[1,2],mean_supplied_gain_pct=x.supplied_gain_pct.mean(),mean_cold_charged_gain_pct=x.charged_gain_pct.mean());(p/'findings.json').write_text(json.dumps(f,indent=2))
