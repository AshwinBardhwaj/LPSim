from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'tools/memory_variants.cuh').read_text();k=s[s.index('__global__ void advanceShared('):]
k=k.replace('advanceShared(MemoryState old','advanceSharedSoA(SharedState old').replace('__shared__ MotionCar local[128]','__shared__ float local[14*128]').replace('local[threadIdx.x]=old.load(i)','CompactState{local,128}.store(threadIdx.x,old.load(i))').replace('old.cache=local','old.soaCache=local')
prefix='''#pragma once
// Field-wise shared cache: consecutive threads address consecutive bank words.
struct SharedState:MemoryState {
 float* soaCache=nullptr;
 __device__ MotionCar load(int id) const {
  if(soaCache){int s=cacheInverse[id]-cacheBase;if(s>=0&&s<cacheCount)return CompactState{soaCache,128}.load(s);}
  return MemoryState::load(id);
 }
};
'''
(r/'tools/shared_soa.cuh').write_text(prefix+k)
