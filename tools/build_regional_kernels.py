from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'src/simulator/scalable_junction_simulator.cu').read_text();compact=(r/'tools/compact_motion.cuh').read_text();out=['#pragma once\n// Generated exact subset kernels and certified road-local temporal motion.\n']
def function(text,start,end):return text[text.index(start):text.index(end,text.index(start))]
for name,end in [('prepareJunctions','__global__ void summarizeJunctions'),('summarizeJunctions','// One thread per vehicle'),('requestJunctionEntry','// Shared admission logic')]:
 k=function(s,'__global__ void '+name,end);brace=k.index('{');k=k[:brace].rstrip()[:-1]+',const int* ids,int count){'+k[brace+1:];k=k.replace(name+'(',name+'Subset(')
 if name=='prepareJunctions':k=k.replace('if(i<n){Car& c=cars[i];','if(i<count){Car& c=cars[ids[i]];')
 else:k=k.replace('if(i>=n)return;','if(i>=count)return;i=ids[i];',1)
 out.append(k)
k=function(compact,'__global__ void extractCompact','__device__ void writeCompact');k=k.replace('extractCompact(const Car* cars, CompactState out)','extractCompactSubset(const Car* cars, CompactState out,const int* ids,int count)').replace('if(i>=out.count)return;','if(i>=count)return;i=ids[i];');out.append(k)
k=compact[compact.index('__global__ void advanceCompact('):];k=k.replace('advanceCompact(','advanceCompactSubset(').replace('const int* movementOpen=nullptr){','const int* movementOpen=nullptr,const int* ids=nullptr,int count=0){').replace('if(i>=n)return;','if(i>=count)return;i=ids[i];',1);out.append(k)
m=(r/'tools/motion_context.cuh').read_text();k=function(m,'__global__ void collectMotionContext','__global__ void resolveMotionAvailability');k=k.replace('collectMotionContext(','collectMotionContextSubset(').replace('int* queried){','int* queried,const int* ids,int count){').replace('if(i>=n)return;','if(i>=count)return;i=ids[i];');out.append(k)
out.append('''
__global__ void makeSubsetLeaderKeys(const Car* cars,const int* work,int count,unsigned long long* keys,int* ids){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=work[k];ids[k]=i;const Car& c=cars[i];if(c.state!=1){keys[k]=~0ULL;return;}unsigned int bits=__float_as_uint(c.pos==0.f?0.f:c.pos),ordered=bits&0x80000000u?~bits:bits^0x80000000u;unsigned int lane=c.edges[c.k]*8+c.lanes[c.k];keys[k]=(static_cast<unsigned long long>(lane)<<32)|ordered;}
void buildSubsetLeaders(LeaderIndex& x,const Car* cars,const int* work,int count){if(!count)return;makeSubsetLeaderKeys<<<(count+127)/128,128>>>(cars,work,count,x.keysIn,x.ids);check(cub::DeviceRadixSort::SortPairs(x.temporary,x.temporaryBytes,x.keysIn,x.keysOut,x.ids,x.order,count));linkLaneLeaders<<<(count+127)/128,128>>>(cars,x.order,x.leaders,count);}
__global__ void makeSubsetRequests(const EntryRequest* requests,const Movement* moves,const int* ids,int count,unsigned long long* keys){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];EntryRequest q=requests[i];keys[k]=q.pass<0?~0ULL:(static_cast<unsigned long long>(moves[q.mid].junction)<<33)|(static_cast<unsigned long long>(q.pass)<<32)|static_cast<unsigned int>(i);}
__global__ void publishLocal(Car* cars,const MotionCar* cache,const int* ids,int count,int h){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<count)writeCompact(cars[ids[k]],cache[h*count+k]);}
''')
body=compact[compact.index('MotionCar c=old.load(i);'):compact.rindex(' writeCompact(next[i],c);')]
a=body.index('  bool open=');b=body.index('  float gap=',a);body=body[:a]+'  bool open=c.k==c.n-1;\n'+body[b:]
a=body.index('  if(leaders)');b=body.index('  float desired=',a)
body=body[:a]+'''  int lead=i+1<old.count&&old.load(i+1).lane==lane?i+1:-1;
  if(lead>=0){MotionCar d=old.load(lead);float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}
'''+body[b:]
fields=['state','k','n','mid','event','denial','departure','pos','v','stopped','entryWait'];load=''.join(f'c.{f}=z.{f};' for f in fields)+'c.edge=z.edges[z.k];c.lane=z.lanes[z.k];c.movement=z.moves[z.k];'
out.append('''
template<int Width> __global__ void localRoadSteps(const Car* cars,const Road* roads,const int* offsets,const int* ids,int selected,int steps,float dt,MotionCar* cache,int groupStart,int groupEnd,bool trajectory=true){
 __shared__ float first[14*128],second[14*128];int localGroup=threadIdx.x/Width,group=groupStart+blockIdx.x*(128/Width)+localGroup,offset=group<groupEnd?offsets[group]:0,count=group<groupEnd?offsets[group+1]-offset:0,i=threadIdx.x%Width;CompactState old{first+localGroup*14*Width,count},next{second+localGroup*14*Width,count};
 if(i<count){const Car& z=cars[ids[offset+i]];MotionCar c;'''+load+'''old.store(i,c);}__syncthreads();
 for(int h=0;h<steps;++h){if(i<count){'''+body+'''next.store(i,c);if(trajectory||h==steps-1)cache[h*selected+offset+i]=c;}__syncthreads();float* tmp=old.values;old.values=next.values;next.values=tmp;}
}
''')
(r/'tools/regional_kernels.cuh').write_text('\n'.join(out))
