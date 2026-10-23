from pathlib import Path
r=Path(__file__).resolve().parents[1]
s=(r/'src/simulator/scalable_junction_simulator.cu').read_text()
s=s.replace('#include "junction_simulator.h"','#include "../src/simulator/junction_simulator.h"').replace('../../tools/','')
s=s.replace('#include <thrust/sort.h>','#include <thrust/sort.h>\n#include <thrust/copy.h>')
# Optional candidate IDs preserve canonical vehicle IDs and arbitration priority.
s=s.replace('JunctionSummary* s,EntryRequest* requests){','JunctionSummary* s,EntryRequest* requests,const int* candidates=nullptr,int candidateCount=0){')
needle='int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const Car& c=cars[i];EntryRequest r'
s=s.replace(needle,'int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=(candidates?candidateCount:n))return;if(candidates)i=candidates[i];const Car& c=cars[i];EntryRequest r')
# Multi-rate kernel is generated directly from compact physics.
k=(r/'tools/compact_motion.cuh').read_text();k=k[k.index('__global__ void advanceCompact('):]
k=k.replace('advanceCompact(','advanceMulti(').replace('const int* movementOpen=nullptr){','const int* movementOpen,float* held,int* previousEdge,int step,int rate,unsigned long long* counts){')
k=k.replace('const int* leaders=nullptr,const int* work=nullptr,int region=-1,const int* connectorHead=nullptr,const int* connectorNext=nullptr,','const int* leaders,const int* work,int region,const int* connectorHead,const int* connectorNext,')
start=k.index('  bool open=');end=k.index('  float speed=',start)
physics=k[start:end]
# Guard evaluated each tick; full dynamics immediately resumes when a leader/connector appears.
pre='''  float horizon=rate*dt;
  bool eligible=leaders[i]<0&&connectorHead[edge*8+lane]<0&&r.length-c.pos>fmaxf(c.v,r.speed)*horizon+SPACING+10.f;
  bool reuse=eligible&&previousEdge[i]==edge&&step%rate!=0;
  float acc,bound=r.length;
  if(reuse){acc=held[i];if(counts)atomicAdd(counts,1ULL);}
  else {
'''
physics=physics.replace('leaderV=0,bound=r.length;','leaderV=0;').replace('float acc=.5570409f','acc=.5570409f')
k=k[:start]+pre+physics+'  held[i]=acc;\n  }\n  previousEdge[i]=eligible?edge:-1;\n'+k[end:]
k=k.replace(' if(c.state==1){',' if(counts&&c.state==1)atomicAdd(counts+1,1ULL);\n if(c.state!=1)previousEdge[i]=-1;\n if(c.state==1){')
(r/'tools/multirate_motion.cuh').write_text('// Generated from compact baseline. Approximate held acceleration, guarded every tick.\n'+k)
s=s.replace('#include "compact_motion.cuh"','#include "compact_motion.cuh"\n#include "multirate_motion.cuh"')
insert='''
struct ValidRequestKey {__host__ __device__ bool operator()(unsigned long long key)const{return key!=~0ULL;}};
__global__ void candidateKeys(const EntryRequest* requests,const Movement* moves,const int* ids,int count,unsigned long long* keys){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];EntryRequest q=requests[i];keys[k]=q.pass<0?~0ULL:(static_cast<unsigned long long>(moves[q.mid].junction)<<33)|(static_cast<unsigned long long>(q.pass)<<32)|static_cast<unsigned int>(i);}
'''
s=s.replace('void runScalableJunctionSimulation(',insert+'\nvoid runScalableJunctionSimulation(')
needle=' int steps=int(std::round(duration/dt));'
init='''
 const std::string experiment=std::getenv("LPSIM_EXPERIMENT")?std::getenv("LPSIM_EXPERIMENT"):"baseline";
 bool trace=experiment=="trace",oracle=experiment=="oracle",current=experiment=="current",timing=std::getenv("LPSIM_TIMING_ONLY");
 int rate=experiment=="multi2"?2:experiment=="multi4"?4:1;
 if(!compactMotion||!parallel||!requestIndex.enabled||!indexed||motionContext.flags!=3)throw std::runtime_error("Experiment requires compact indexed parallel baseline");
 std::ofstream traceFile;if(trace)traceFile.open("request_trace.bin",std::ios::binary);
 std::vector<EntryRequest> hostRequests(n);
 std::vector<std::vector<int>> masks;
 if(oracle){std::ifstream masksFile(std::getenv("LPSIM_ORACLE_MASK"),std::ios::binary);int windows;masksFile.read((char*)&windows,4);masks.resize(windows);for(auto& v:masks){int count;masksFile.read((char*)&count,4);v.resize(count);masksFile.read((char*)v.data(),count*4);}if(!masksFile)throw std::runtime_error("Invalid masks");}
 int* candidateIds=device(std::vector<int>(n));int candidateCount=0;
 unsigned long long* packed=device(std::vector<unsigned long long>(n));
 float* held=device(std::vector<float>(n));int* previousEdge=device(std::vector<int>(n,-1));
 unsigned long long* counters=timing?nullptr:device(std::vector<unsigned long long>(2));
 double assignmentMs=0;unsigned long long submitted=0;
'''
s=s.replace(needle,needle+init)
old='advanceCompact<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,dt,mode,exclusive,leaders,nullptr,-1,motionContext.head,motionContext.links,motionContext.available);'
s=s.replace(old,'if(rate==1){'+old+'}else advanceMulti<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,dt,mode,exclusive,leaders,nullptr,-1,motionContext.head,motionContext.links,motionContext.available,held,previousEdge,step,rate,counters);')
old='auto start=std::chrono::steady_clock::now();check(cudaEventRecord(ev0));'
new='''auto start=std::chrono::steady_clock::now();
  if(oracle&&step%8==0){auto begin=std::chrono::steady_clock::now();const auto& ids=masks.at(step/8);candidateCount=ids.size();if(candidateCount)check(cudaMemcpy(candidateIds,ids.data(),candidateCount*4,cudaMemcpyHostToDevice));assignmentMs+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();}
  check(cudaEventRecord(ev0));'''
s=s.replace(old,new)
old='requestJunctionEntry<<<(n+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq);'
s=s.replace(old,'if(!oracle||candidateCount)requestJunctionEntry<<<((oracle?candidateCount:n)+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq,oracle?candidateIds:nullptr,candidateCount);')
old='if(requestIndex.enabled){requestIndex.build(dq,dm);'
new='''if(requestIndex.enabled){
    if(oracle||current){
     check(cudaMemset(requestIndex.begin,0,nj*4));check(cudaMemset(requestIndex.end,0,nj*4));
     int count=candidateCount;
     if(current){makeJunctionRequestKeys<<<(n+127)/128,128>>>(dq,dm,packed,n);auto first=thrust::device_pointer_cast(packed);count=thrust::copy_if(first,first+n,thrust::device_pointer_cast(requestIndex.keys),ValidRequestKey())-thrust::device_pointer_cast(requestIndex.keys);}
     else if(count)candidateKeys<<<(count+127)/128,128>>>(dq,dm,candidateIds,count,requestIndex.keys);
     if(count){thrust::sort(thrust::device_pointer_cast(requestIndex.keys),thrust::device_pointer_cast(requestIndex.keys)+count);findJunctionRequestRanges<<<(count+127)/128,128>>>(requestIndex.keys,count,requestIndex.begin,requestIndex.end);}
     submitted+=count;
    }else {requestIndex.build(dq,dm);submitted+=n;}
'''
s=s.replace(old,new)
# Trace outside timing. Timing repetitions omit host validation/output, preserving identical GPU work.
old='std::swap(a,b);check(cudaMemcpy(cars.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));'
new='''std::swap(a,b);
  if(trace){check(cudaMemcpy(hostRequests.data(),dq,n*sizeof(EntryRequest),cudaMemcpyDeviceToHost));std::vector<int> ids;for(int i=0;i<n;++i)if(hostRequests[i].pass>=0)ids.push_back(i);int count=ids.size();traceFile.write((char*)&count,4);traceFile.write((char*)ids.data(),count*4);}
  if(timing)continue;
  check(cudaMemcpy(cars.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));'''
s=s.replace(old,new)
# Equivalent zero-violation boundary checks in linear time (counts are violating connectors).
start=s.index('  for(const auto& c:cars)if(c.state==2){const auto& m=moves[c.mid];for(');end=s.index('  if((step+1)%2==0)',start)
s=s[:start]+'''  std::vector<float> nearest(nr*8,1.e30f),furthest(nr*8,-1.e30f);for(const auto& c:cars)if(c.state==1){int key=c.edges[c.k]*8+c.lanes[c.k];nearest[key]=std::min(nearest[key],c.pos);furthest[key]=std::max(furthest[key],c.pos);}
  for(const auto& c:cars)if(c.state==2){const auto& m=moves[c.mid];gapFailures+=m.length-c.pos+nearest[m.out*8+m.outLane]<SPACING-.001f;gapFailures+=roads[m.in].length-furthest[m.in*8+m.inLane]+c.pos<SPACING-.001f;}
'''+s[end:]
needle=' int completed=0,active=0,pending=0;'
new=''' check(cudaMemcpy(cars.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));
 std::ofstream finalState("final.bin",std::ios::binary);finalState.write((char*)cars.data(),n*sizeof(Car));
 unsigned long long hostCounters[2]={0,0};if(counters)check(cudaMemcpy(hostCounters,counters,16,cudaMemcpyDeviceToHost));
 std::ofstream experimentMetrics("experiment.json");experimentMetrics<<"{\\"rate\\":"<<rate<<",\\"reused_acceleration_updates\\":"<<hostCounters[0]<<",\\"road_updates\\":"<<hostCounters[1]<<",\\"submitted_keys\\":"<<submitted<<",\\"assignment_ms\\":"<<assignmentMs<<"}\\n";
'''
s=s.replace(needle,new+needle)
(r/'tools/gating_multirate_simulator.cu').write_text('// Generated experimental runner; production implementation unchanged.\n'+s)
