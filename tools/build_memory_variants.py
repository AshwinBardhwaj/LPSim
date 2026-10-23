from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'tools/compact_motion.cuh').read_text();k=s[s.index('__global__ void advanceCompact('):]
prefix='''// Generated motion variants; the equations remain the compact baseline equations.
#pragma once
struct MemoryState {
 CompactState storage;const int* order=nullptr;const int* inverse=nullptr;
 const int* laneMap=nullptr;const int* best=nullptr;
 const MotionCar* cache=nullptr;const int* cacheInverse=nullptr;int cacheBase=0,cacheCount=0;
 __device__ MotionCar load(int id) const {
  if(cache){int s=cacheInverse[id]-cacheBase;if(s>=0&&s<cacheCount)return cache[s];}
  return storage.load(inverse?inverse[id]:id);
 }
 __device__ void store(int id,const MotionCar& c) const {storage.store(inverse?inverse[id]:id,c);}
};
__global__ void inverseSlots(const int* order,int* inverse,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)inverse[order[i]]=i;}
__global__ void extractMemory(const Car* cars,MemoryState dst){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dst.storage.count)return;const Car& a=cars[i];MotionCar c;
'''
fields=['state','k','n','mid','event','denial','departure','pos','v','stopped','entryWait'];prefix+='\n'.join(f'c.{x}=a.{x};' for x in fields)
prefix+='\nc.edge=a.edges[a.k];c.lane=a.lanes[a.k];c.movement=a.moves[a.k];dst.store(i,c);}\n'
k=k.replace('advanceCompact(CompactState old','advanceMemory(MemoryState old').replace('if(i>=n)return;if(work','if(i>=n)return;if(old.order)i=old.order[i];if(work')
k=k.replace('int chosenConnector=-1;','int chosenConnector=-1;int entry=old.laneMap?old.laneMap[edge*8+lane]:-1;int best=entry>=0?old.best[entry]:-2;')
k=k.replace('for(int j=connectorHead?connectorHead[edge*8+lane]:0;', 'for(int j=best!=-2?best:(connectorHead?connectorHead[edge*8+lane]:0);')
k=k.replace('j=connectorHead?connectorNext[j]:j+1)', 'j=best!=-2?-1:(connectorHead?connectorNext[j]:j+1))')
shared=k.replace('advanceMemory(MemoryState old','advanceShared(MemoryState old')
old='int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;if(old.order)i=old.order[i];'
new='''__shared__ MotionCar local[128];int slot=blockIdx.x*blockDim.x+threadIdx.x;
 int i=slot<n?old.order[slot]:0;if(slot<n)local[threadIdx.x]=old.load(i);__syncthreads();if(slot>=n)return;
 old.cache=local;old.cacheBase=blockIdx.x*128;old.cacheCount=min(128,n-old.cacheBase);'''
assert old in shared;shared=shared.replace(old,new)
prefix+='''
__global__ void buildNearestConnector(const Car* cars,const int* head,const int* links,const int* laneKeys,int* best,int count){
 int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;int key=laneKeys[slot],chosen=-1;float pos=1.e30f;
 for(int j=head[key];j>=0;j=links[j]){float p=cars[j].pos;if(p<pos||(p==pos&&(chosen<0||j<chosen))){pos=p;chosen=j;}}
 best[slot]=chosen;
}
'''
(r/'tools/memory_variants.cuh').write_text(prefix+k+shared)
