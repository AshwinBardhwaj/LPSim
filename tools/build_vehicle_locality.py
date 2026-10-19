#!/usr/bin/env python3
from pathlib import Path
import subprocess,json,hashlib
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/vehicle-allocation-locality-20261018';A.mkdir(parents=True,exist_ok=True)
s=(R/'src/simulator/scalable_junction_simulator.cu').read_text();k=s[s.index('__global__ void advance('):s.index('// Deterministic junction arbitration')]
k=k.replace('__global__ void advance(', 'template<bool Instrument> __global__ void advanceTasks(').replace('const Car* old,Car* next,int n,','const Car* old,Car* next,int n,const int* taskIds,unsigned long long* ticks,')
k=k.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;', 'int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=n)return;int i=taskIds[slot];unsigned long long started=0;if constexpr(Instrument)started=clock64();')
k=k.replace('next[i]=c;', 'next[i]=c;if constexpr(Instrument)ticks[slot]=clock64()-started;')
(A/'advance_tasks.cuh').write_text(k)
cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_87','-std=c++17',str(R/'tools/vehicle_locality_replay.cu'),'-o',str(A/'replay')]
(A/'build.json').write_text(json.dumps(dict(command=cmd,original_source_sha256=hashlib.sha256(s.encode()).hexdigest(),generated_kernel_sha256=hashlib.sha256(k.encode()).hexdigest()),indent=2))
with (A/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
