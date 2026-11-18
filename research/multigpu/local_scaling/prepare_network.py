#!/usr/bin/env python3
"""Full Berkeley research model and reproducible state-cost experiment."""
import argparse,csv,json,math,random,hashlib,subprocess,os,time
from pathlib import Path
from collections import defaultdict
import networkx as nx
from shapely.geometry import LineString
ROOT=Path(__file__).resolve().parents[3]
ART=ROOT/'artifacts/local-scaling-20261028'
NET=ROOT/'data/networks/sf_partial_local_scaling'
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 assert not (NET/'model.txt').exists(), 'Do not overwrite a prepared study'
 NET.mkdir(parents=True,exist_ok=True)
 nodes={int(x['osmid']):x for x in csv.DictReader((ROOT/'data/networks/sf_partial/nodes.csv').open())}
 edges=list(csv.DictReader((ROOT/'data/networks/sf_partial/edges.csv').open()))
 def xy(n):return ((float(nodes[n]['x'])+122.29)*111320*math.cos(math.radians(37.87)),(float(nodes[n]['y'])-37.87)*111320)
 groups=nx.Graph();groups.add_nodes_from(nodes)
 for e in edges:
  if float(e['length'])<25:groups.add_edge(int(e['u']),int(e['v']))
 components=sorted(nx.connected_components(groups),key=min);owner={n:i for i,ns in enumerate(components) for n in ns}
 roads=[];outgoing=defaultdict(list)
 for e in edges:
  u,v=int(e['u']),int(e['v']);a,b=owner[u],owner[v]
  if a==b:continue
  p,q=xy(u),xy(v);length=math.dist(p,q)
  if length<=16:continue
  tangent=[(q[k]-p[k])/length for k in (0,1)]
  r=dict(id=len(roads),source_edge=int(e['uniqueid']),a=a,b=b,lanes=min(8,max(1,int(e['lanes']))),length=float(e['length'])-16,speed=float(e['speed_mph'])*.44704,tangent=tangent,start=[p[k]+8*tangent[k] for k in (0,1)],end=[q[k]-8*tangent[k] for k in (0,1)])
  assert r['length']>0 and 1<=r['lanes']<=8
  roads.append(r);outgoing[a].append(r)
 def lane_point(r,l,end):
  offset=(l+.5-r['lanes']/2)*3.2;dx,dy=r['tangent'];p=r['end' if end else 'start'];return [p[0]+dy*offset,p[1]-dx*offset]
 moves=[];byjunction=defaultdict(list);graph=nx.DiGraph()
 for r in roads:
  for l in range(r['lanes']):graph.add_node((r['id'],l))
 for a in roads:
  ax,ay=a['tangent']
  for b in outgoing[a['b']]:
   bx,by=b['tangent'];dot=ax*bx+ay*by;cross=ax*by-ay*bx
   if b['b']==a['a'] or dot<-.75:continue
   turn=1 if cross>.4 else -1 if cross<-.4 else 0
   pairs=[(0,0)] if turn==1 else [(a['lanes']-1,b['lanes']-1)] if turn==-1 else [(l,l) for l in range(min(a['lanes'],b['lanes']))]
   for il,ol in pairs:
    p0,p3=lane_point(a,il,True),lane_point(b,ol,False);d=math.dist(p0,p3)
    p1=[p0[0]+ax*d*.45,p0[1]+ay*d*.45];p2=[p3[0]-bx*d*.45,p3[1]-by*d*.45]
    path=[[(1-t)**3*p0[k]+3*(1-t)**2*t*p1[k]+3*(1-t)*t*t*p2[k]+t**3*p3[k] for k in (0,1)] for t in [i/40 for i in range(41)]]
    length=LineString(path).length
    # Very short connectors are conservatively lengthened in the motion model.
    m=dict(id=len(moves),junction=a['b'],incoming=a['id'],outgoing=b['id'],in_lane=il,out_lane=ol,turn=turn,axis=int(abs(ax)>=abs(ay)),length=max(8.,length),rtor=int(turn==-1),path=path)
    moves.append(m);byjunction[a['b']].append(m['id']);graph.add_edge((a['id'],il),(b['id'],ol),weight=b['length']/b['speed']+length/7,mid=m['id'])
 shapes=[LineString(m['path']).buffer(1.2) for m in moves];conflicts=[]
 for mids in byjunction.values():
  for i in mids:
   a=moves[i]
   for j in mids:
    b=moves[j]
    if (a['incoming'],a['in_lane'])==(b['incoming'],b['in_lane']) or (a['outgoing'],a['out_lane'])==(b['outgoing'],b['out_lane']) or shapes[i].intersects(shapes[j]):conflicts.append((i,j))
 with (NET/'model.txt').open('w') as f:
  f.write(f'{len(roads)} {len(moves)} {len(components)}\n')
  for r in roads:f.write(' '.join(str(r[k]) for k in ('a','b','lanes','length','speed'))+'\n')
  for m in moves:f.write(' '.join(str(m[k]) for k in ('junction','incoming','outgoing','in_lane','out_lane','turn','axis','length','rtor'))+'\n')
  f.write(f'SPARSE {len(conflicts)}\n');f.writelines(f'{a} {b}\n' for a,b in conflicts)
 dump(NET/'geometry.json',dict(roads=roads,movements=moves,groups=[sorted(x) for x in components],conflicts=conflicts))

 # Generate reproducible lane-continuous stochastic routes, biased mildly to main roads.
 rng=random.Random(2810); routes=[]; starts=list(graph.nodes)
 for _ in range(12000):
  start=rng.choice(starts);path=[start];seen={start};target=rng.randint(8,28)
  for step in range(target):
   options=[q for q in graph.successors(path[-1]) if q not in seen]
   if not options:break
   nxt=rng.choices(options,weights=[math.sqrt(max(1,roads[q[0]]['speed'])*roads[q[0]]['lanes']) for q in options])[0]
   path.append(nxt);seen.add(nxt)
  if len(path)>=2:routes.append(dict(edges=[x[0] for x in path],lanes=[x[1] for x in path],moves=[graph[a][b]['mid'] for a,b in zip(path,path[1:])]))
 dump(NET/'route_pool.json',routes)
 dump(NET/'provenance.json',dict(source='data/networks/sf_partial',source_hashes={n:sha(ROOT/'data/networks/sf_partial'/n) for n in ['nodes.csv','edges.csv']},roads=len(roads),junctions=len(components),movements=len(moves),routes=len(routes),synthetic=True,notes='Same geometry/connector construction as Berkeley research model; Lane counts clamped to supported 1..8 range; stochastic lane-continuous routes, not calibrated OD demand. No cross-network validation claim until exact physical tests pass.'))
 print(len(roads),len(moves),len(components),len(routes),flush=True)
if __name__=='__main__':prepare()
