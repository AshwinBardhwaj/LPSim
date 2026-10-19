#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,subprocess
R=Path(__file__).resolve().parents[1];A=R/'artifacts/current/berkeley-block-imbalance-20261019';A.mkdir(parents=True,exist_ok=True)
s=(R/'src/simulator/scalable_junction_simulator.cu').read_text();k=s[s.index('__global__ void advance('):s.index('// Deterministic junction arbitration')]
k=k.replace('__global__ void advance(', '__global__ void advanceBlocks(').replace('const Car* old,Car* next,int n,','const Car* old,Car* next,int n,unsigned long long* starts,unsigned long long* ends,unsigned int* sms,')
k=k.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;', 'if(threadIdx.x==0){unsigned long long ts;unsigned int sm;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));starts[blockIdx.x]=ts;sms[blockIdx.x]=sm;}__syncthreads();int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;')
k=k.replace('next[i]=c;', 'next[i]=c;__syncthreads();if(threadIdx.x==0){unsigned long long ts;asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ts));ends[blockIdx.x]=ts;}')
(A/'advance_blocks.cuh').write_text(k)
cmd=['/usr/local/cuda/bin/nvcc','-O3','-lineinfo','-arch=sm_87','-std=c++17','--ptxas-options=-v',str(R/'tools/block_imbalance_replay.cu'),'-o',str(A/'replay')];(A/'build.json').write_text(json.dumps(dict(command=cmd,source_sha256=hashlib.sha256(s.encode()).hexdigest(),instrumented_kernel_sha256=hashlib.sha256(k.encode()).hexdigest()),indent=2))
with (A/'build.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
