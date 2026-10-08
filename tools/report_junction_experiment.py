#!/usr/bin/env python3
"""Render the junction-behavior PDF, figures and recorded movement-aware playback."""
import csv,json,math,os,bisect,shutil,hashlib,zipfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/lpsim-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/junction_platoons_20261007';OUT=ROOT/'reports/san-pablo-platoons-20261007';NET=ROOT/'data/networks/san_pablo_lane_preserving'
def read(p):return list(csv.DictReader(p.open()))
def lane_path(r,l):
 out=[];offset=(l+.5-r['lanes']/2)*3.2
 for i,p in enumerate(r['path']):
  a,b=r['path'][max(0,i-1)],r['path'][min(len(r['path'])-1,i+1)];dx,dy=b[0]-a[0],b[1]-a[1];d=math.hypot(dx,dy);out.append([p[0]+dy/d*offset,p[1]-dx/d*offset])
 return out

def main():
 model=json.loads((NET/'junction_model.json').read_text());roads=model['roads'];moves=model['movements'];prov=json.loads((NET/'provenance.json').read_text());mainj=prov['main_junction'];results={}
 for p in sorted(ART.glob('*/metrics.json')):
  assert json.loads((p.parent/'validation.json').read_text())['passed'];results[p.parent.name]=json.loads(p.read_text())
 assert len(results)==11
 (OUT/'figures').mkdir(parents=True,exist_ok=True);(OUT/'web').mkdir(exist_ok=True)
 pages=[]
 def save(fig,name,caption):
  fig.text(.06,.025,caption,fontsize=8,color='#536779');fig.savefig(OUT/'figures'/f'{name}.png',dpi=160);fig.savefig(OUT/'figures'/f'{name}.svg');pdf.savefig(fig);plt.close(fig);pages.append((name,caption))
 with PdfPages(OUT/'San-Pablo-Physical-Junctions.pdf',metadata={'Title':'San Pablo–University: physical junction behavior','Author':'LPSim','Subject':'Synthetic control assumptions, CUDA simulation and validation'}) as pdf:
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.91,'San Pablo–University',fontsize=29,weight='bold');fig.text(.06,.85,'Lane-preserving turns and safely spaced platoons',fontsize=20,color='#087e8b')
  lines=['Prepared October 7, 2026','21 physical junction groups · 59 external road links · 132 lane connectors','Opt-in CUDA junction backend · 0.25 s timestep · 600 simulated seconds','','Stops occur on entry approaches, not on internal OSM links.','Through/right and protected-left phases are separate.','Right-on-red requires a full stop, permission and a conflicting-traffic gap.','Each vehicle reserves a downstream storage slot before entering.','Straight: same lane; right turn: rightmost lane; left turn: leftmost lane.','','8 controlled fixtures + 3 network scenarios passed their checks.','Normal OD locations are restricted to designated outer-network groups.','No stopped connector samples or simultaneous conflicting movements.','','This is a more explicit, physically constrained experiment model.','It is not a calibrated reproduction of Berkeley traffic or field signal timing.']
  for i,line in enumerate(lines):fig.text(.065,.77-i*.039,line,fontsize=12)
  save(fig,'01-overview','Source: cached OpenStreetMap network. Original occupancy-fix recordings remain unchanged.')
  fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.12,top=.9)
  for r in roads:
   for l in range(r['lanes']):p=lane_path(r,l);ax.plot(*zip(*p),color='#a7b8c1',lw=2)
  for m in moves:
   if m['junction']==mainj:ax.plot(*zip(*m['path']),color={-1:'#087e8b',0:'#5373ad',1:'#d38e26'}[m['turn']],alpha=.5)
  ax.set(xlim=(-65,65),ylim=(-65,65),aspect='equal',xlabel='Meters east of intersection',ylabel='Meters north',title='One physical junction: entry stop lines and lane-to-lane connectors')
  for r in roads:
   if r['b']==mainj:
    p=r['path'][-1];ax.scatter(*p,marker='s',s=45,color='#ba3c32');ax.annotate('Stop line',p,xytext=(7,7),textcoords='offset points',fontsize=9)
  save(fig,'02-junction','Red squares: entry approach ends. Connector curves are schematic, not surveyed lane markings. © OpenStreetMap contributors.')
  fig,ax=plt.subplots(figsize=(11.7,8.3));fig.subplots_adjust(bottom=.14,top=.85)
  labels=['NS through/right','NS protected left','EW through/right','EW protected left'];starts=[0,24,36,60];lengths=[20,8,20,8]
  for y,(st,d) in enumerate(zip(starts,lengths)):
   ax.broken_barh([(0,72)],(y-.3,.6),facecolors='#eed3d0');ax.broken_barh([(st,d)],(y-.3,.6),facecolors='#3ca66b');ax.broken_barh([(st+d,3)],(y-.3,.6),facecolors='#e5bb42')
  ax.set(yticks=range(4),yticklabels=labels,xlim=(0,72),xlabel='Seconds within the synthetic 72-second cycle',title='20-second through greens, protected lefts, yellow and all-red clearance')
  ax.grid(axis='x',alpha=.2)
  save(fig,'03-signals','Right-on-red: permitted scenario movements only, stop ≥2 s, predicted conflicting-arrival gap ≥4 s; blocked during all-red clearance.')
  fig,axs=plt.subplots(3,1,figsize=(11.7,8.3),sharex=True);fig.subplots_adjust(bottom=.12,top=.9,hspace=.35)
  for ax,n in zip(axs,(20,400,1600)):
   rs=read(ART/f'network_{n}'/'timeseries.csv');t=[float(r['time'])/60 for r in rs];ax.plot(t,[int(r['active']) for r in rs],label='Active',color='#087e8b');ax.plot(t,[int(r['stopped']) for r in rs],label='Stopped',color='#cf553c');ax.plot(t,[int(r['pending']) for r in rs],label='Waiting to enter',color='#a5a8b0');ax.set_title(f'{n:,} requested trips',loc='left');ax.legend(loc='upper right');ax.set_ylabel('Vehicles')
  axs[-1].set_xlabel('Elapsed simulation minutes');fig.suptitle('Queues remain on roads outside junction interiors',fontsize=19)
  save(fig,'04-congestion','Vehicle fronts have a 6.5 m minimum same-lane separation (4.5 m vehicle + 2 m gap), checked every simulation step.')
  fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');fig.subplots_adjust(bottom=.14,top=.86)
  rows=[]
  for name,m in results.items():rows.append([name,str(m['completed'])+'/'+str(m['trips']),m['right_on_red_entries'],m['downstream_blocked_decisions'],m['yield_blocked_decisions']])
  table=ax.table(cellText=rows,colLabels=['Scenario','Completed','Right on red','Space denials','Yield denials'],loc='center',cellLoc='center',colWidths=[.28,.18,.18,.18,.18]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.3);ax.set_title('Movement tests and demand results',fontsize=20)
  save(fig,'05-results','Denials count repeated timestep decisions, not distinct vehicles. Held-red and blocked-exit fixtures intentionally do not complete.')
  fig=plt.figure(figsize=(11.7,8.3));fig.text(.06,.91,'What this result establishes—and what it does not',fontsize=22,weight='bold')
  blocks=[('Passed at every 0.25-second step','No invalid state, sub-6.5 m same-lane spacing, conflicting occupied movement\ncorridors, or stopped cars on internal connectors. Movement entry logs are audited.'),('Controlled tests','Right on red after stopping; yield to a conflicting green movement; no entry\nwith a blocked exit; red through/left hold; green through/left completion.'),('Safety assumptions','Different conflicting movements remain exclusive; identical movements can\nfollow at ≥2 s admission headway with downstream storage for every vehicle. Pedestrians/cyclists and emergency braking are not modeled.'),('Road and turning assumptions','Stop lines are cropped 8 m from OSM nodes; lane paths use 3.2 m spacing.\nThrough movements preserve lane index; right/left turns enter rightmost/leftmost lane.'),('Matched comparison in this report','Exclusive and platoon modes use identical lane geometry, route demand and timing.\nOctober 6 results use different routes and connectors and are not matched baselines.'),('Field validation still required','The 72 s plan, right-on-red permissions and left phases are synthetic assumptions.\nActual Berkeley signs, crosswalks, signal plans and observed flows were not collected.')]
  for i,(title,body) in enumerate(blocks):y=.8-i*.117;fig.text(.065,y,title,fontsize=12,weight='bold');fig.text(.065,y-.03,body,fontsize=11,va='top')
  save(fig,'06-scope','Phase structure reference: FHWA Traffic Signal Timing Manual, Chapters 4–5. No claim of traffic-engineering certification.')
  from junction_report_metrics import add_pages
  add_pages(ROOT,ART,OUT,results,pdf,save)
 # Playback includes actual connector positions and movement-specific recorded indications.
 (OUT/'web/network.json').write_text(json.dumps(roads));(OUT/'web/movements.json').write_text(json.dumps(moves));meta={}
 lane_geometry={(r['id'],l):lane_path(r,l) for r in roads for l in range(r['lanes'])}
 offsets_cache={}
 def position(path,distance,total):
  key=id(path)
  if key not in offsets_cache:
   offsets=[0.]
   for a,b in zip(path,path[1:]):offsets.append(offsets[-1]+math.dist(a,b))
   offsets_cache[key]=offsets
  offsets=offsets_cache[key]
  s=max(0,min(distance/total*offsets[-1],offsets[-1]));j=max(0,min(len(path)-2,bisect.bisect_right(offsets,s)-1));d=offsets[j+1]-offsets[j];f=(s-offsets[j])/d if d else 0
  return [path[j][k]+f*(path[j+1][k]-path[j][k]) for k in (0,1)]
 for name in ('network_20','network_400','network_1600','exclusive_20','exclusive_400','exclusive_1600'):
  exclusive=name.startswith('exclusive_');case='network_'+name.split('_')[-1];source=ART/'exclusive'/case if exclusive else ART/case
  summary=json.loads((source/'metrics.json').read_text())
  folder=OUT/'web'/name;folder.mkdir(exist_ok=True);chunks={};queues={};samples=0
  for r in csv.DictReader((source/'states.csv').open()):
   t=float(r['time']);frame=round(t*2);state=int(r['state']);edge=int(r['edge']);lane=int(r['lane']);mid=int(r['movement']);p=float(r['pos']);v=float(r['speed'])
   path=moves[mid]['path'] if state==2 else lane_geometry[edge,lane];length=moves[mid]['length'] if state==2 else roads[edge]['length'];x,y=position(path,p,length)
   chunks.setdefault(int(t//20),{}).setdefault(frame,[]).append([int(r['id']),round(x,3),round(y,3),round(v,4),edge,lane,state,mid]);samples+=1
   if v<.5:queues.setdefault(frame,{}).setdefault((edge,lane),0);queues[frame][edge,lane]+=1
  for b,data in chunks.items():(folder/f'{b}.json').write_text(json.dumps(data,separators=(',',':')))
  lights={}
  for r in csv.DictReader((source/'signals.csv').open()):lights.setdefault(round((float(r['time'])+.25)*2),[0]*len(moves))[int(r['movement'])]=int(r['state'])
  stats={round(float(r['time'])*2):[int(r['active']),int(r['stopped']),max(queues.get(round(float(r['time'])*2),{}).values(),default=0),int(r['completed']),int(r['pending'])] for r in read(source/'timeseries.csv')}
  assert sum(s[0] for s in stats.values())==samples
  meta[name]=dict(chunks=sorted(chunks),frames=1200,signals=lights,stats=stats,summary=summary,samples=samples,main_junction=mainj)
  print(name,samples,'playback samples',flush=True)
 (OUT/'web/manifest.json').write_text(json.dumps(meta,separators=(',',':')));shutil.copy2(ROOT/'viz/junction_viewer.html',OUT/'web/index.html')
 (OUT/'summary.json').write_text(json.dumps(results,indent=2))
 (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>San Pablo junction behavior</title><style>body{font:17px system-ui;max-width:1100px;margin:35px auto;background:#f5f8fa;color:#17324d}img{width:100%}section{margin:35px 0}a{color:#087e8b}</style><h1>San Pablo physical-junction experiment</h1><p><a href="San-Pablo-Physical-Junctions.pdf">PDF report</a> · <a href="web/">New traffic visualization</a></p>'+''.join(f'<section><img src="figures/{n}.png" alt="{n}"><p>{c}</p></section>' for n,c in pages))
 print('Report ready:',OUT)
if __name__=='__main__':main()
