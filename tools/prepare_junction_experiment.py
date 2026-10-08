#!/usr/bin/env python3
"""Build physical junctions, lane connectors and boundary-only demand from cached OSM."""
import csv,json,math,random,shutil,os
from pathlib import Path
import networkx as nx
from shapely import wkt
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'data/networks/san_pablo_university_small';NET=ROOT/'data/networks/san_pablo_lane_preserving';ART=ROOT/'artifacts/junction_platoons_20261007'
def rows(p):return list(csv.DictReader(p.open()))
def main():
 NET.mkdir(parents=True,exist_ok=True);ART.mkdir(parents=True,exist_ok=True)
 for n in ('nodes.csv','edges.csv','source.graphml'):shutil.copy2(SRC/n,NET/n)
 nodes={int(r['index']):r for r in rows(SRC/'nodes.csv')};edges=rows(SRC/'edges.csv');g=nx.Graph();g.add_nodes_from(nodes)
 for e in edges:
  if float(e['length'])<25:g.add_edge(int(e['u']),int(e['v']))
 groups=sorted(nx.connected_components(g),key=min);group={n:i for i,ns in enumerate(groups) for n in ns}
 osm=nx.read_graphml(SRC/'source.graphml',node_type=int,force_multigraph=True);roads=[]
 def xy(p):return ((p[0]+122.29216)*111320*math.cos(math.radians(37.8691)),(p[1]-37.8691)*111320)
 for e in edges:
  u,v=int(e['u']),int(e['v']);a,b=group[u],group[v]
  if a==b:continue
  data=next(iter(osm[int(e['osmid_u'])][int(e['osmid_v'])].values()));raw=list(wkt.loads(data['geometry']).coords) if 'geometry'in data else [(float(nodes[u]['x']),float(nodes[u]['y'])),(float(nodes[v]['x']),float(nodes[v]['y']))]
  line=LineString([xy(p) for p in raw]);p0=xy((float(nodes[u]['x']),float(nodes[u]['y'])))
  if math.dist(line.coords[0],p0)>math.dist(line.coords[-1],p0):line=LineString(list(line.coords)[::-1])
  # Crop the road before entry and beyond exit; connectors fill the physical box.
  path=[list(line.interpolate(8+(line.length-16)*i/30).coords[0]) for i in range(31)]
  roads.append(dict(id=len(roads),osm_edge=int(e['uniqueid']),a=a,b=b,lanes=int(e['lanes']),length=line.length-16,speed=float(e['speed_mph'])*.44704,path=path,name=str(data.get('name','Street'))))
 def lane_path(r,l):
  pts=[];offset=(l+.5-r['lanes']/2)*3.2
  for i,p in enumerate(r['path']):
   a,b=r['path'][max(0,i-1)],r['path'][min(30,i+1)];dx,dy=b[0]-a[0],b[1]-a[1];d=math.hypot(dx,dy)
   pts.append([p[0]+dy/d*offset,p[1]-dx/d*offset])
  return pts
 moves=[];lookup={};jg=nx.DiGraph()
 for r in roads:jg.add_edge(r['a'],r['b'],weight=r['length'],edge=r['id'])
 for a in roads:
  av=[a['path'][-1][i]-a['path'][-2][i] for i in (0,1)];an=math.hypot(*av);av=[v/an for v in av]
  axis=0 if abs(av[1])>abs(av[0]) else 1
  for b in roads:
   if b['a']!=a['b'] or b['b']==a['a']:continue
   bv=[b['path'][1][i]-b['path'][0][i] for i in (0,1)];bn=math.hypot(*bv);bv=[v/bn for v in bv];cross=av[0]*bv[1]-av[1]*bv[0];dot=sum(x*y for x,y in zip(av,bv))
   if dot<-.75:continue
   turn=1 if cross>.4 else -1 if cross<-.4 else 0
   for il in range(a['lanes']):
    if turn==1 and il!=0 or turn==-1 and il!=a['lanes']-1:continue
    for ol in range(b['lanes']):
     if (turn==0 and il!=ol) or (turn==1 and ol!=0) or (turn==-1 and ol!=b['lanes']-1):continue
     p0=lane_path(a,il)[-1];p3=lane_path(b,ol)[0];d=math.dist(p0,p3);p1=[p0[i]+av[i]*d*.45 for i in (0,1)];p2=[p3[i]-bv[i]*d*.45 for i in (0,1)]
     path=[]
     for k in range(41):
      t=k/40;path.append([(1-t)**3*p0[i]+3*(1-t)**2*t*p1[i]+3*(1-t)*t*t*p2[i]+t**3*p3[i] for i in (0,1)])
     offsets=[0.]
     for p,q in zip(path,path[1:]):offsets.append(offsets[-1]+math.dist(p,q))
     m=dict(id=len(moves),junction=a['b'],incoming=a['id'],outgoing=b['id'],in_lane=il,out_lane=ol,turn=turn,axis=axis,length=offsets[-1],path=path,offsets=offsets,rtor=int(turn==-1))
     lookup[a['id'],b['id'],il,ol]=m['id'];moves.append(m)
 shapes=[LineString(m['path']).buffer(1.2) for m in moves]
 conflicts=[[int(a['junction']==b['junction'] and (a['incoming']==b['incoming'] and a['in_lane']==b['in_lane'] or a['outgoing']==b['outgoing'] and a['out_lane']==b['out_lane'] or shapes[i].intersects(shapes[j]))) for j,b in enumerate(moves)] for i,a in enumerate(moves)]
 model=dict(roads=roads,movements=moves,junctions=[sorted(ns) for ns in groups],conflicts=conflicts,phase_seconds=dict(through_green=20,yellow=3,all_red=1,left_green=8),vehicle_length=4.5,minimum_gap=2.,rtor_stop_seconds=2.,rtor_gap_seconds=4.,rtor_signage='assumed permitted for right movements; not field verified',pedestrians='not modeled; no pedestrian calls in these scenarios')
 (NET/'junction_model.json').write_text(json.dumps(model,indent=2))
 with (NET/'model.txt').open('w') as f:
  f.write(f'{len(roads)} {len(moves)} {len(groups)}\n')
  for r in roads:f.write(f"{r['a']} {r['b']} {r['lanes']} {r['length']} {r['speed']}\n")
  for m in moves:f.write(' '.join(str(m[k]) for k in ('junction','incoming','outgoing','in_lane','out_lane','turn','axis','length','rtor'))+'\n')
  for row in conflicts:f.write(' '.join(map(str,row))+'\n')
 # Boundary groups, not interior graph nodes, are the only OD candidates.
 centers={i:xy((sum(float(nodes[n]['x']) for n in ns)/len(ns),sum(float(nodes[n]['y']) for n in ns)/len(ns))) for i,ns in enumerate(groups)}
 boundary=sorted(n for n in jg if abs(centers[n][0])>220 or abs(centers[n][1])>220)
 choices=[]
 for a in boundary:
  for b in boundary:
   if a==b:continue
   try:route=nx.shortest_path(jg,a,b,weight='weight')
   except nx.NetworkXNoPath:continue
   es=[jg[u][v]['edge'] for u,v in zip(route,route[1:])]
   if len(es)>=3 and all(any(m['incoming']==u and m['outgoing']==v for m in moves) for u,v in zip(es,es[1:])):choices.append(es)
 assert choices, boundary
 def lane_options(es):
  paths=[([lane],[]) for lane in range(roads[es[0]]['lanes'])]
  for u,v in zip(es,es[1:]):
   paths=[(ls+[m['out_lane']],ms+[m['id']]) for ls,ms in paths for m in moves if m['incoming']==u and m['outgoing']==v and m['in_lane']==ls[-1]]
  return paths
 choices=[es for es in choices if lane_options(es)]
 assert choices, 'No lane-continuous routes'
 def make_trip(es,i,dep):
  options=lane_options(es);lanes,ms=options[i%len(options)]
  return dict(departure=dep,edges=es,lanes=lanes,moves=ms)
 cases=[]
 for n in (20,400,1600):
  rng=random.Random(42);trips=[make_trip(rng.choice(choices),i,rng.uniform(0,180)) for i in range(n)];trips.sort(key=lambda t:t['departure']);cases.append((f'network_{n}',trips,0))
 # Controlled movements use approach origins away from the box, for targeted assertions.
 center=group[next(n for n,r in nodes.items() if r['osmid']=='53116953')]
 targets={turn:next(m for m in moves if m['junction']==center and m['turn']==turn) for turn in (-1,0,1)}
 for turn,label,mode in [(-1,'right_on_red',1),(0,'through_red',1),(1,'left_red',1),(0,'through_green',2),(1,'left_green',2)]:
  m=targets[turn];trip=dict(departure=0,edges=[m['incoming'],m['outgoing']],lanes=[m['in_lane'],m['out_lane']],moves=[m['id']],initial_pos=roads[m['incoming']]['length']);cases.append((label,[trip],mode))
 # An occupied downstream road must prevent a red right turn from entering the box.
 m=targets[-1]
 downstream=next(x for x in moves if x['incoming']==m['outgoing'] and x['in_lane']==m['out_lane'] and x['turn']!=-1)
 def seeded(m,pos):return dict(departure=0,edges=[m['incoming'],m['outgoing']],lanes=[m['in_lane'],m['out_lane']],moves=[m['id']],initial_pos=pos)
 length=roads[m['outgoing']]['length']
 blocked=[seeded(m,roads[m['incoming']]['length'])]+[seeded(downstream,length-6.5*k) for k in range(int(length/6.5)+1)]
 cases.append(('blocked_exit',blocked,1))
 # Red right turn yields to an already admitted conflicting green movement.
 red=next(m for m in moves if m['junction']==center and m['turn']==-1 and m['axis']==1 and any(conflicts[m['id']][x['id']] and x['axis']==0 and x['turn']==0 for x in moves))
 green=next(x for x in moves if conflicts[red['id']][x['id']] and x['axis']==0 and x['turn']==0)
 cases.append(('right_yields',[seeded(green,roads[green['incoming']]['length']),seeded(red,roads[red['incoming']]['length'])],0))
 # Saturated approach fixture demonstrates multiple safely spaced cars on one connector.
 m=targets[0]
 platoon=[seeded(m,roads[m['incoming']]['length']-6.5*k) for k in range(12)]
 cases.append(('same_movement_platoon',platoon,2))
 for name,trips,mode in cases:
  folder=ART/name;folder.mkdir(exist_ok=True)
  with (folder/'scenario.txt').open('w') as f:
   f.write(f'{NET / "model.txt"}\n{len(trips)} 600 0.25 {mode}\n')
   for t in trips:f.write(f"{t['departure']} {len(t['edges'])} "+' '.join(map(str,t['edges']))+' '+' '.join(map(str,t['lanes']))+' '+' '.join(map(str,t['moves']))+' '+str(t.get('initial_pos',-1))+'\n')
  (folder/'demand.json').write_text(json.dumps(trips));(folder/'case.json').write_text(json.dumps(dict(name=name,trips=len(trips),mode=mode,duration=600,dt=.25,seed=42)))
 (NET/'provenance.json').write_text(json.dumps(dict(source=str(SRC.relative_to(ROOT)),internal_nodes_grouped=[sorted(ns) for ns in groups],boundary_groups=boundary,main_junction=center,signal_plan='Synthetic 72 s cycle; not measured Berkeley timing',attribution='© OpenStreetMap contributors'),indent=2))
 print(len(roads),'roads',len(moves),'movements',len(groups),'physical junctions; OD boundary groups',boundary)
if __name__=='__main__':main()
