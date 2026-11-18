"""Paper-style baselines, not a claim to reproduce unspecified original settings.
Route-edge visits are known demand, never future simulated occupancy. METIS
balances vertex count; Leiden modularity communities are spatially clustered.
No silent fallback when a required partitioning library is missing.
"""
import json, math, time
import numpy as np

def partition(geometry, cars, policy, parts=3, seed=2710):
    started=time.perf_counter()
    roads=geometry['roads']; n=max(max(r['a'],r['b']) for r in roads)+1
    if not 1<=parts<=n: raise ValueError('Invalid partition count')
    points=[[] for _ in range(n)]
    for r in roads:
        points[r['a']].append(r['start']);points[r['b']].append(r['end'])
    xy=np.array([np.mean(p,axis=0) if p else [0,0] for p in points])
    visits=[0]*len(roads)
    for c in cars:
        for rid in c[6:6+c[2]]:
            if not 0<=rid<len(roads):raise ValueError('Invalid route road')
            visits[rid]+=1
    weights={}
    for rid,r in enumerate(roads):
        a,b=sorted((r['a'],r['b']))
        if a!=b and visits[rid]:weights[a,b]=weights.get((a,b),0)+visits[rid]
    used=sorted({v for e in weights for v in e}); remap={v:i for i,v in enumerate(used)}
    if len(used)<parts:raise ValueError('Demand visits fewer nodes than partitions')
    edges=sorted(weights); details={}
    if parts==1:owners=[0]*n
    elif policy=='geography':
        owners=np.minimum(parts-1,((xy[:,0]-xy[:,0].min())/max(float(np.ptp(xy[:,0])),1)*parts).astype(int)).tolist()
    elif policy=='paper_balanced':
        import pymetis
        adj=[[] for _ in used]
        for a,b in edges:adj[remap[a]].append((remap[b],weights[a,b]));adj[remap[b]].append((remap[a],weights[a,b]))
        xadj=[0];neighbors=[];ew=[]
        for row in adj:
            for b,w in sorted(row):neighbors.append(b);ew.append(w)
            xadj.append(len(neighbors))
        cut,assignment=pymetis.part_graph(parts,xadj=xadj,adjncy=neighbors,eweights=ew,options=pymetis.Options(seed=seed,ufactor=30),recursive=False)
        details={'solver':'METIS multilevel k-way','imbalance_factor':1.03,'balance_weight':'unit visited node','metis_cut':cut}
        owners=fill_unvisited(n,used,list(assignment),xy,parts)
    elif policy=='paper_community':
        import igraph as ig, leidenalg
        graph=ig.Graph(n=len(used),edges=[(remap[a],remap[b]) for a,b in edges],directed=False)
        communities=leidenalg.find_partition(graph,leidenalg.ModularityVertexPartition,weights=[weights[e] for e in edges],seed=seed,n_iterations=-1)
        groups=list(communities)
        if len(groups)<parts:raise ValueError('Too few Leiden communities; do not silently split them')
        centers=np.array([xy[[used[i] for i in group]].mean(axis=0) for group in groups])
        labels=kmeans(centers,parts,seed)
        assignment=[-1]*len(used)
        for group,p in zip(groups,labels):
            for i in group:assignment[i]=int(p)
        owners=fill_unvisited(n,used,assignment,xy,parts)
        details={'solver':'Leiden modularity + geographic Lloyd k-means','communities':len(groups),'modularity':communities.quality()}
    else:raise ValueError(policy)
    if set(owners)!=set(range(parts)):raise ValueError('Empty partition')
    return owners,dict(policy=policy,parts=parts,seed=seed,construction_ms=1000*(time.perf_counter()-started),edge_weight='full known-demand route traversal counts, undirected summed',future_trajectory_access=False,unvisited='nearest demand-partition centroid',node_counts=[owners.count(p) for p in range(parts)],route_weight_cut=sum(w for (a,b),w in weights.items() if owners[a]!=owners[b]),cut_roads=sum(owners[r['a']]!=owners[r['b']] for r in roads),**details)

def fill_unvisited(n,used,assignment,xy,parts):
    owners=[-1]*n
    for node,p in zip(used,assignment):owners[node]=p
    centers=np.array([xy[[v for v in used if owners[v]==p]].mean(axis=0) for p in range(parts)])
    if not np.isfinite(centers).all():raise ValueError('Empty demand partition')
    for v in range(n):
        if owners[v]<0:owners[v]=int(np.argmin(((centers-xy[v])**2).sum(axis=1)))
    return owners

def kmeans(x,k,seed):
    rng=np.random.default_rng(seed); centers=[x[rng.integers(len(x))]]
    while len(centers)<k:
        distances=np.min(((x[:,None,:]-np.array(centers)[None,:,:])**2).sum(axis=2),axis=1)
        centers.append(x[int(np.argmax(distances))])
    centers=np.array(centers);last=None
    for _ in range(300):
        labels=np.argmin(((x[:,None,:]-centers[None,:,:])**2).sum(axis=2),axis=1)
        if set(labels)!=set(range(k)):raise ValueError('Empty spatial community cluster')
        if last is not None and np.array_equal(last,labels):return labels
        last=labels.copy();centers=np.array([x[labels==p].mean(axis=0) for p in range(k)])
    return labels
