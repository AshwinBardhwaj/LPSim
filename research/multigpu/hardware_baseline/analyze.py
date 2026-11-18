"""Summarize measured hardware samples; preserve raw timing and units."""
from pathlib import Path
import argparse,csv,hashlib,json,statistics,tarfile
import numpy as np
p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('report',type=Path);a=p.parse_args();a.report.mkdir(parents=True,exist_ok=True)
archives=list(a.bundle.glob('hardware-*.tar.gz'))
if len(archives)!=1:raise RuntimeError('Expected exactly one returned hardware archive')
raw=a.bundle/'raw';raw.mkdir(exist_ok=True)
with tarfile.open(archives[0]) as tar:
    for member in tar.getmembers():
        target=(raw/member.name).resolve()
        if raw.resolve() not in target.parents or not (member.isfile() or member.isdir()):raise RuntimeError('Unsafe archive member')
        if member.isdir():target.mkdir(parents=True,exist_ok=True)
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(tar.extractfile(member).read())
d=next(raw.glob('*/inventory.json')).parent
for name,expected in json.loads((d/'sha256.json').read_text()).items():
    if hashlib.sha256((d/name).read_bytes()).hexdigest()!=expected:raise RuntimeError('Output hash mismatch '+name)
inv=json.loads((d/'inventory.json').read_text());status=json.loads((d/'status.json').read_text())
with (d/'samples.csv').open() as f:samples=list(csv.DictReader(f))
if not status['passing'] or any(s['valid']!='1' for s in samples):raise RuntimeError('Hardware validation failed')
groups={}
for s in samples:
    key=(s['test'],int(s['src']),int(s['dst']),int(s['bytes']));groups.setdefault(key,[]).append(s)
summary=[]
for (test,src,dst,size),rows in sorted(groups.items()):
    out=dict(test=test,src=src,dst=dst,bytes=size,samples=len(rows),iterations_per_sample=int(rows[0]['iterations']))
    for field in ['device_us','wall_us','gbps','tflops']:
        values=[float(r[field]) for r in rows]
        for name,q in [('p10',10),('median',50),('p90',90)]:out[field+'_'+name]=float(np.percentile(values,q))
        out[field+'_min']=min(values);out[field+'_max']=max(values)
    summary.append(out)
with (a.report/'summary.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
(a.report/'summary.json').write_text(json.dumps({'inventory':inv,'status':status,'sample_count':len(samples),'summary':summary,'raw_directory':str(d.resolve())},indent=2)+'\n')
print('PASS verified artifact hashes and',len(samples),'raw samples;',len(summary),'summary rows')
for s in summary:
    if s['test'].startswith('gemm') or s['test']=='hbm_triad' or (s['test'] in ['peer_batch','peer_roundtrip'] and s['bytes'] in [64,67108864]):
        print(s['test'],s['src'],s['dst'],s['bytes'],'device_us',round(s['device_us_median'],3),'wall_us',round(s['wall_us_median'],3),'GB/s',round(s['gbps_median'],3),'TF/s',round(s['tflops_median'],3))
