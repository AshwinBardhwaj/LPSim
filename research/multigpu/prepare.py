"""Freeze inputs and three interchangeable partition policies; no GPU required."""
from pathlib import Path
import argparse, hashlib, json, math, random, shutil, struct, subprocess
ROOT = Path(__file__).resolve().parents[2]
CAR = struct.Struct('<102i5f')
POLICIES = ('geography', 'vehicle_count', 'communication_graph')

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def read_snapshot(path):
    raw = Path(path).read_bytes()
    n, t = struct.unpack_from('<if', raw)
    if n < 1 or len(raw) != 8+n*CAR.size: raise ValueError('Snapshot size/ABI mismatch')
    return t, [CAR.unpack_from(raw, 8+i*CAR.size) for i in range(n)]

def partition(geometry, cars, policy):
    roads = geometry['roads']; nj = max(max(r['a'], r['b']) for r in roads)+1
    coords = [[] for _ in range(nj)]
    for r in roads:
        coords[r['a']].append(r['start']); coords[r['b']].append(r['end'])
    xs = [sum(p[0] for p in points)/len(points) if points else 0 for points in coords]
    lo, hi = min(xs), max(xs)
    geographic = [min(2, int(3*(x-lo)/max(hi-lo, 1))) for x in xs]
    weights = [0]*nj
    for c in cars:
        if c[0] in (1, 2): weights[roads[c[6+c[1]]]['b']] += 1
    fallback = sum(weights) == 0
    if fallback: weights = [1]*nj
    edges = {}; adjacency = [[] for _ in range(nj)]
    # Static byte proxy: 24-byte reservation per lane plus 60-byte road state per
    # lane. Not a forecast or measured bandwidth. Both directions are combined.
    for r in roads:
        key = tuple(sorted((r['a'], r['b'])))
        if key[0] != key[1]: edges[key] = edges.get(key, 0)+84*r['lanes']
    for (a,b), w in edges.items(): adjacency[a].append((b,w)); adjacency[b].append((a,w))
    def loads(assignment): return [sum(w for j,w in enumerate(weights) if assignment[j] == p) for p in range(3)]
    if policy == 'geography': assignment = geographic
    elif policy == 'vehicle_count':
        assignment = [0]*nj; load = [0]*3; members = [0]*3
        for j in sorted(range(nj), key=lambda j: (-weights[j], j)):
            p = min(range(3), key=lambda p: (load[p], members[p], p))
            assignment[j] = p; load[p] += weights[j]; members[p] += 1
    elif policy == 'communication_graph':
        assignment = geographic[:]; load = loads(assignment)
        # Deterministic balanced graph-cut heuristic, not METIS or an optimum.
        # Penalize load squared; same observed population weights as count policy.
        scale = sum(edges.values())/max(sum(weights)**2, 1)*6
        for sweep in range(30):
            changed = False
            for j in sorted(range(nj), key=lambda j: (-sum(w for _,w in adjacency[j]), j)):
                old = assignment[j]; weight = weights[j]
                costs = []
                for p in range(3):
                    cut = sum(w*((assignment[b] != p)-(assignment[b] != old)) for b,w in adjacency[j])
                    balance = 0 if p == old else scale*((load[old]-weight)**2+(load[p]+weight)**2-load[old]**2-load[p]**2)
                    costs.append(cut+balance)
                best = min(range(3), key=lambda p: (costs[p], p != old, p))
                if costs[best] < -1e-9:
                    load[old] -= weight; load[best] += weight; assignment[j] = best; changed = True
            if not changed: break
    else: raise ValueError(policy)
    meta = dict(policy=policy, partitions=3, population_weights='unit_junction_fallback' if fallback else 'observed_road_and_connector', loads=loads(assignment), junction_counts=[assignment.count(p) for p in range(3)], cut_roads=sum(assignment[r['a']] != assignment[r['b']] for r in roads), cut_lane_count=sum(r['lanes'] for r in roads if assignment[r['a']] != assignment[r['b']]), cut_byte_proxy=sum(w for (a,b),w in edges.items() if assignment[a] != assignment[b]), online_repartitioning=False)
    if any(x == 0 for x in meta['junction_counts']): raise ValueError('Empty partition')
    return assignment, meta

def write_policies(folder, geometry, snapshot):
    _, cars = read_snapshot(snapshot); result = {}
    for name in POLICIES:
        owners, meta = partition(geometry, cars, name)
        target = folder/(name+'.txt'); target.write_text('\n'.join(map(str,owners))+'\n')
        meta['sha256'] = sha(target); (folder/(name+'.json')).write_text(json.dumps(meta, indent=2)+'\n'); result[name] = meta
    return result

def source_closure(start):
    import re
    found = set()
    def visit(p):
        p=p.resolve()
        if p in found: return
        found.add(p)
        for rel in re.findall(r'^\s*#include\s+"([^"]+)"', p.read_text(), re.M):
            target=p.parent/rel
            if target.exists(): visit(target)
    for p in start: visit(p)
    return found

def prepare(output):
    output = output.resolve(); output.mkdir(parents=True, exist_ok=False)
    network = ROOT/'data/networks/berkeley_junction_study'
    geometry=json.loads((network/'geometry.json').read_text()); pool=json.loads((network/'route_pool.json').read_text())
    (output/'inputs').mkdir(); shutil.copytree(network, output/'inputs/network')
    old=ROOT/'artifacts/current/compact-motion-20261022/runs'
    cases=[]
    for count,time in ((2048,0),(8192,180),(16384,180)):
        name=f'reference_{count}_t{time}'; d=output/'inputs'/name; d.mkdir()
        source=old/f'distributed_{count}_seed1901'/f'snapshot_{time}.bin'
        shutil.copy2(source,d/'snapshot.bin')
        policies=write_policies(d,geometry,d/'snapshot.bin')
        cases.append(dict(name=name,split='historical_correctness',population=count,start=time,snapshot=f'inputs/{name}/snapshot.bin',policies=policies,sha256=sha(d/'snapshot.bin')))
    roads=geometry['roads']
    weights=[1+.30*sum(max(0,roads[e]['lanes']-1)+max(0,roads[e]['speed']/11.176-1) for e in route['edges'])/len(route['edges']) for route in pool]
    # Freeze new train/validation/test seeds now. These are synthetic, mildly
    # arterial-biased demands, not calibrated OD estimates or unseen networks.
    for split,seed in [('train',2701),('validation',2702),('test',2703)]:
        for pattern in ['distributed','concentrated','bursty']:
            for count in [2048,8192,16384]:
                rng=random.Random(f'{seed}:{pattern}:{count}'); name=f'{split}_{pattern}_{count}'; d=output/'inputs'/name; d.mkdir()
                ws=weights[:]
                if pattern=='concentrated':
                    center=roads[pool[seed%len(pool)]['edges'][0]]['start']
                    ws=[w*(1+2*math.exp(-sum((x-y)**2 for x,y in zip(roads[r['edges'][0]]['start'],center))/(2*700**2))) for w,r in zip(ws,pool)]
                routes=rng.choices(pool, weights=ws, k=count); rows=[]
                for route in routes:
                    depart=rng.uniform(0,180)
                    if pattern=='bursty': depart=min(179.999,max(0,rng.choice([20,80,140])+rng.gauss(0,7)))
                    rows.append((depart,route))
                rows.sort(key=lambda x:x[0]); raw=bytearray(struct.pack('<if',count,0.0))
                for depart,r in rows:
                    arrays=[]
                    for key in ['edges','lanes','moves']: arrays+=r[key]+[0]*(32-len(r[key]))
                    raw.extend(CAR.pack(0,0,len(r['edges']),-1,0,0,*arrays,depart,0,0,0,0))
                (d/'snapshot.bin').write_bytes(raw)
                # All pending: initial count policy explicitly uses fallback.
                policies=write_policies(d,geometry,d/'snapshot.bin')
                cases.append(dict(name=name,split=split,seed=seed,pattern=pattern,population=count,start=0,snapshot=f'inputs/{name}/snapshot.bin',sha256=sha(d/'snapshot.bin'),policies=policies))
    manifest=dict(schema=1,created='2026-10-27',horizon_s=600,timestep_s=.25,controller_mode=0,following=True,launch_threads=128,partitions=3,cases=cases,network_hash=sha(network/'model.txt'),demand_notes='Synthetic routes sampled from the existing 2048-route pool; modest lane/speed bias, seeded departures. No calibration claim. All splits share Berkeley topology; external-network generalization remains untested.',larger_network='No larger validated junction model available locally; do not fabricate one or claim cross-network validation.')
    (output/'workloads.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();prepare(a.output)
