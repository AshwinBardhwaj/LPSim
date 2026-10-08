#include "junction_simulator.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <cstdlib>

namespace {
constexpr int MAX_ROUTE=32;
constexpr float SPACING=6.5f; // 4.5 m vehicle + 2 m stopped gap
struct Road { int from,to,lanes; float length,speed; };
struct Movement { int junction,in,out,inLane,outLane,turn,axis;float length;int rtor; };
struct Car { int state=0,k=0,n=0,mid=-1,event=0,denial=0;int edges[MAX_ROUTE],lanes[MAX_ROUTE],moves[MAX_ROUTE];float departure=0,pos=0,v=0,stopped=0,entryWait=0; };
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>T* device(const std::vector<T>& a){T* p;check(cudaMalloc(&p,a.size()*sizeof(T)));check(cudaMemcpy(p,a.data(),a.size()*sizeof(T),cudaMemcpyHostToDevice));return p;}
// 72 s: NS through 20 + yellow 3 + all red 1; NS left 8+3+1;
// EW through 20+3+1; EW left 8+3+1. No new entry on yellow.
__host__ __device__ int indication(const Movement& m,float t,int mode){
 if(mode==1)return 0;if(mode==2)return 2;
 float p=fmodf(t,72.f);if(m.axis==1)p=fmodf(p+36.f,72.f);
 if(m.turn==1){if(p>=24 && p<32)return 2;if(p>=32 && p<35)return 1;}
 else {if(p<20)return 2;if(p<23)return 1;}
 return 0;
}
__host__ __device__ bool clearance(float t,int mode){if(mode)return false;float p=fmodf(t,36.f);return (p>=23&&p<24)||p>=35;}
__global__ void advance(const Car* old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;Car c=old[i];c.event=0;c.denial=0;
 if(c.state==1){const int edge=c.edges[c.k],lane=c.lanes[c.k];const Road r=roads[edge];
  bool open=c.k==c.n-1;
  if(c.k+1<c.n){int mid=c.moves[c.k];const Movement m=moves[mid];open=indication(m,t,mode)==2;
   if(open)for(int j=0;j<n;++j){const Car& d=old[j];
    if((d.state==2&&conflict[mid*nm+d.mid]&&(exclusive||mid!=d.mid))||
       (d.state==1&&d.edges[d.k]==m.out&&d.lanes[d.k]==m.outLane&&d.pos<SPACING+8.f)){open=false;break;}}
  }
  float gap=open?1.e6f:r.length-c.pos+2,leaderV=0,bound=r.length;
  for(int j=0;j<n;++j){if(j==i)continue;const Car& d=old[j];if(d.state==1&&d.edges[d.k]==edge&&d.lanes[d.k]==lane&&d.pos>c.pos){float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}
  for(int j=0;j<n;++j){const Car& d=old[j];if(d.state==2&&moves[d.mid].in==edge&&moves[d.mid].inLane==lane){
   float distance=r.length-c.pos+d.pos;if(distance-4.5f<gap){gap=distance-4.5f;leaderV=d.v;}bound=fminf(bound,r.length+d.pos-SPACING);
  }}
  float desired=1.38075f+fmaxf(0.f,c.v*.5433028f+c.v*(c.v-leaderV)/(2*sqrtf(.5570409f*2.902058f)));
  float acc=.5570409f*(1-powf(c.v/r.speed,4)-powf(desired/fmaxf(.05f,gap),2));
  float speed=fmaxf(0.f,c.v+acc*dt);float proposed=c.pos+.5f*(c.v+speed)*dt;
  if(proposed>bound){proposed=fmaxf(c.pos,bound);speed=0;}
  c.pos=proposed;c.v=speed;
  if(r.length-c.pos<.1f && c.v<.3f){c.pos=r.length;c.v=0;}
  c.stopped=(c.v<.1f&&r.length-c.pos<.11f)?c.stopped+dt:0;
  if(c.k==c.n-1&&r.length-c.pos<.11f)c.state=3;
 }
 next[i]=c;
}
// Deterministic junction arbitration on the GPU; all road propagation is parallel.
// Conflicting movements remain exclusive; identical movements may form platoons.
// Every admitted vehicle reserves one downstream storage slot until it exits.
__global__ void junctionStep(Car* cars,int n,const Road* roads,const Movement* moves,
                            const int* conflict,int nm,float t,float dt,int mode,bool exclusive){
 if(blockIdx.x||threadIdx.x)return;
 for(int i=0;i<n;++i){Car& c=cars[i];if(c.state!=2)continue;const Movement m=moves[c.mid];c.pos+=c.v*dt;
  if(c.pos>=m.length){c.pos-=m.length;c.k++;c.state=1;c.event=3;c.stopped=0;c.mid=-1;}}
 // Summarize occupied exits and approaching green movements once per step.
 float nearest[512],arrival[512],tail[512];int busy[512],reserved[512],owner[512];
 for(int k=0;k<512;++k){nearest[k]=1.e9f;arrival[k]=1.e9f;busy[k]=0;reserved[k]=0;owner[k]=-1;tail[k]=1.e9f;}
 for(int j=0;j<n;++j){const Car& d=cars[j];
  if(d.state==2){busy[d.mid]++;tail[d.mid]=fminf(tail[d.mid],d.pos);int key=moves[d.mid].out*4+moves[d.mid].outLane;reserved[key]++;owner[key]=d.mid;}
  if(d.state==1){int key=d.edges[d.k]*4+d.lanes[d.k];nearest[key]=fminf(nearest[key],d.pos);
   if(d.k+1<d.n){int mid=d.moves[d.k];if(indication(moves[mid],t,mode)==2)arrival[mid]=fminf(arrival[mid],(roads[d.edges[d.k]].length-d.pos)/fmaxf(1.f,d.v));}}
 }
 // Green traffic has priority over right-on-red candidates.
 for(int pass=0;pass<2;++pass)for(int i=0;i<n;++i){Car& c=cars[i];
  if(c.state!=1||c.k+1>=c.n||roads[c.edges[c.k]].length-c.pos>.11f)continue;
  int mid=c.moves[c.k];const Movement m=moves[mid];int light=indication(m,t,mode);
  bool red=light==0&&m.turn==-1&&m.rtor&&c.stopped>=2.f&&!clearance(t,mode);
  if((pass==0&&light!=2)||(pass==1&&(!red||light==2)))continue;
  int exitKey=m.out*4+m.outLane;
  if(reserved[exitKey]&&(exclusive||owner[exitKey]!=mid)){c.denial=1;continue;}
  float speed=m.turn==0?7.f:4.f;
  if(!exclusive&&tail[mid]<fmaxf(SPACING,2.f*speed)){c.denial=1;continue;}
  if(fminf(nearest[exitKey],roads[m.out].length)<SPACING*(reserved[exitKey]+1)+8.f){c.denial=2;continue;}
  bool blocked=false;
  for(int j=0;j<nm;++j)if(conflict[mid*nm+j]){
   if(busy[j]&&(exclusive||j!=mid)){blocked=true;c.denial=1;break;}
   if(pass==1&&j!=mid&&arrival[j]<4.f){blocked=true;c.denial=3;break;}
  }
  if(blocked)continue;
  c.entryWait=c.stopped;c.state=2;c.mid=mid;c.pos=0;c.v=m.turn==0?7.f:4.f;c.event=pass==1?2:1;c.stopped=0;
  busy[mid]++;tail[mid]=0;reserved[exitKey]++;owner[exitKey]=mid;
 }
 // Spawn at the outer-network road entrance, never on an internal connector.
 for(int i=0;i<n;++i){Car& c=cars[i];if(c.state!=0||c.departure>t)continue;
  int key=c.edges[0]*4+c.lanes[0];
  if(nearest[key]>=SPACING&&!reserved[key]){c.state=1;c.pos=0;c.v=0;nearest[key]=0;}
 }
}
// All junction-stage reads are separated from writes by kernel boundaries.
// Each receiving lane belongs to exactly one junction (Road::from).
struct JunctionSummary {
 float nearest[512],arrival[512],tail[512];
 int busy[512],reserved[512],owner[512],spawn[512];
};
struct EntryRequest { int mid,pass,blockedIndex,blockedReason; };
__device__ void minPositive(float* target,float value){atomicMin(reinterpret_cast<int*>(target),__float_as_int(value));}
__global__ void prepareJunctions(Car* cars,int n,const Movement* moves,float dt,JunctionSummary* s){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<512){s->nearest[i]=s->arrival[i]=s->tail[i]=1.e9f;s->busy[i]=s->reserved[i]=0;s->owner[i]=-1;s->spawn[i]=n;}
 if(i<n){Car& c=cars[i];if(c.state==2){const Movement m=moves[c.mid];c.pos+=c.v*dt;
  if(c.pos>=m.length){c.pos-=m.length;c.k++;c.state=1;c.event=3;c.stopped=0;c.mid=-1;}}}
}
__global__ void summarizeJunctions(const Car* cars,int n,const Road* roads,const Movement* moves,float t,int mode,JunctionSummary* s){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const Car& c=cars[i];
 if(c.state==2){int key=moves[c.mid].out*4+moves[c.mid].outLane;atomicAdd(s->busy+c.mid,1);minPositive(s->tail+c.mid,c.pos);atomicAdd(s->reserved+key,1);atomicMax(s->owner+key,c.mid);}
 if(c.state==1){minPositive(s->nearest+c.edges[c.k]*4+c.lanes[c.k],c.pos);
  if(c.k+1<c.n){int mid=c.moves[c.k];if(indication(moves[mid],t,mode)==2)minPositive(s->arrival+mid,(roads[c.edges[c.k]].length-c.pos)/fmaxf(1.f,c.v));}}
 if(c.state==0&&c.departure<=t)atomicMin(s->spawn+c.edges[0]*4+c.lanes[0],i);
}
// One thread per vehicle evaluates signal/stop eligibility and existing conflicts.
__global__ void requestJunctionEntry(const Car* cars,int n,const Road* roads,const Movement* moves,const int* conflict,int nm,float t,int mode,bool exclusive,const JunctionSummary* s,EntryRequest* requests){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const Car& c=cars[i];EntryRequest r{-1,-1,nm,0};
 if(c.state==1&&c.k+1<c.n&&roads[c.edges[c.k]].length-c.pos<=.11f){
  int mid=c.moves[c.k];const Movement m=moves[mid];int light=indication(m,t,mode);
  bool red=light==0&&m.turn==-1&&m.rtor&&c.stopped>=2.f&&!clearance(t,mode);
  if(light==2||red){r.mid=mid;r.pass=light==2?0:1;
   for(int j=0;j<nm;++j)if(conflict[mid*nm+j]){
    if(s->busy[j]&&(exclusive||j!=mid)){r.blockedIndex=j;r.blockedReason=1;break;}
    if(r.pass==1&&j!=mid&&s->arrival[j]<4.f){r.blockedIndex=j;r.blockedReason=3;break;}
   }
  }
 }
 requests[i]=r;
}
// One arbiter per junction, with independent junctions executing concurrently.
// Vehicle-ID priority and green-before-red ordering match the serial reference.
__global__ void arbitrateJunctions(Car* cars,int n,const Road* roads,const Movement* moves,const int* conflict,int nm,int nj,bool exclusive,JunctionSummary* s,const EntryRequest* requests){
 int junction=blockIdx.x*blockDim.x+threadIdx.x;if(junction>=nj)return;
 // Only newly granted movements need testing; existing conflicts were evaluated
 // by the requesting vehicle's thread. Local IDs are bounded by movement count.
 int granted[512],count=0;
 for(int pass=0;pass<2;++pass)for(int i=0;i<n;++i){
  const EntryRequest r=requests[i];if(r.pass!=pass||moves[r.mid].junction!=junction)continue;
  Car& c=cars[i];if(c.state!=1)continue;const Movement m=moves[r.mid];int key=m.out*4+m.outLane;
  if(s->reserved[key]&&(exclusive||s->owner[key]!=r.mid)){c.denial=1;continue;}
  float speed=m.turn==0?7.f:4.f;
  if(!exclusive&&s->tail[r.mid]<fmaxf(SPACING,2.f*speed)){c.denial=1;continue;}
  if(fminf(s->nearest[key],roads[m.out].length)<SPACING*(s->reserved[key]+1)+8.f){c.denial=2;continue;}
  int blocked=r.blockedIndex,reason=r.blockedReason;
  for(int k=0;k<count;++k){int j=granted[k];if(j<=blocked&&conflict[r.mid*nm+j]&&(exclusive||j!=r.mid)){blocked=j;reason=1;}}
  if(blocked<nm){c.denial=reason;continue;}
  c.entryWait=c.stopped;c.state=2;c.mid=r.mid;c.pos=0;c.v=speed;c.event=pass==1?2:1;c.stopped=0;
  s->tail[r.mid]=0;s->reserved[key]++;s->owner[key]=r.mid;granted[count++]=r.mid;
 }
}
// One writer per receiving lane. Winner IDs came from the immutable pending set;
// arbitration has finished publishing receiving-space reservations before this launch.
__global__ void spawnJunctionVehicles(Car* cars,int n,int nr,const JunctionSummary* s){
 int key=blockIdx.x*blockDim.x+threadIdx.x;if(key>=nr*4)return;int id=s->spawn[key];
 if(id<n&&s->nearest[key]>=SPACING&&!s->reserved[key]){Car& c=cars[id];c.state=1;c.pos=0;c.v=0;}
}

}
void runJunctionSimulation(const char* scenario){
 bool parallel=!(std::getenv("LPSIM_JUNCTION_PARALLEL")&&std::string(std::getenv("LPSIM_JUNCTION_PARALLEL"))=="0");
 bool exclusive=std::getenv("LPSIM_JUNCTION_EXCLUSIVE")&&std::string(std::getenv("LPSIM_JUNCTION_EXCLUSIVE"))=="1";
 std::ifstream f(scenario);std::string model;int n,mode;float duration,dt;
 if(!(f>>model>>n>>duration>>dt>>mode)||n<1||dt<=0)throw std::runtime_error("Invalid junction scenario");
 std::ifstream in(model);int nr,nm,nj;if(!(in>>nr>>nm>>nj)||nr<1||nm<1||nr>128||nm>512||nj<1)throw std::runtime_error("Invalid junction model");
 std::vector<Road> roads(nr);std::vector<Movement> moves(nm);std::vector<int> conflicts(nm*nm);
 for(auto& r:roads)if(!(in>>r.from>>r.to>>r.lanes>>r.length>>r.speed)||r.length<=0||r.lanes<1||r.lanes>4)throw std::runtime_error("Invalid road");
 for(auto& m:moves)if(!(in>>m.junction>>m.in>>m.out>>m.inLane>>m.outLane>>m.turn>>m.axis>>m.length>>m.rtor)||m.length<=0)throw std::runtime_error("Invalid movement");
 for(auto& v:conflicts)in>>v;if(!in)throw std::runtime_error("Invalid conflict matrix");
 // Partition ownership is required to avoid cross-junction reservation races.
 for(const auto& m:moves)if(m.in<0||m.in>=nr||m.out<0||m.out>=nr||m.junction<0||m.junction>=nj||roads[m.in].to!=m.junction||roads[m.out].from!=m.junction)throw std::runtime_error("Invalid junction ownership");
 for(int i=0;i<nm;++i)for(int j=0;j<nm;++j)if(conflicts[i*nm+j]&&moves[i].junction!=moves[j].junction)throw std::runtime_error("Cross-junction conflict requires shared arbitration");
 std::vector<Car> cars(n);
 for(auto& c:cars){f>>c.departure>>c.n;if(c.n<2||c.n>MAX_ROUTE)throw std::runtime_error("Invalid route length");for(int k=0;k<c.n;++k)f>>c.edges[k];for(int k=0;k<c.n;++k)f>>c.lanes[k];for(int k=0;k<c.n-1;++k)f>>c.moves[k];float initial;f>>initial;if(initial>=0){c.state=1;c.pos=initial;}
  for(int k=0;k<c.n;++k)if(c.edges[k]<0||c.edges[k]>=nr||c.lanes[k]<0||c.lanes[k]>=roads[c.edges[k]].lanes)throw std::runtime_error("Invalid route lane");
  for(int k=0;k<c.n-1;++k){int id=c.moves[k];if(id<0||id>=nm||moves[id].in!=c.edges[k]||moves[id].out!=c.edges[k+1]||moves[id].inLane!=c.lanes[k]||moves[id].outLane!=c.lanes[k+1])throw std::runtime_error("Invalid route movement");}}
 if(!f)throw std::runtime_error("Incomplete scenario");
 auto* dr=device(roads);auto* dm=device(moves);auto* dc=device(conflicts);auto* a=device(cars);auto* b=device(cars);
 JunctionSummary* ds;EntryRequest* dq;check(cudaMalloc(&ds,sizeof(JunctionSummary)));check(cudaMalloc(&dq,n*sizeof(EntryRequest)));
 std::ofstream traj("states.csv"),events("events.csv"),signals("signals.csv"),series("timeseries.csv");
 traj<<"time,id,state,edge,lane,movement,pos,speed\n";events<<"time,id,event,movement,turn,axis,light,wait_seconds\n";signals<<"time,movement,state\n";series<<"time,active,stopped,in_junction,completed,pending\n";
 traj<<std::setprecision(9);events<<std::setprecision(9);
 unsigned long long gapFailures=0,conflictFailures=0,invalid=0,insideStops=0,redTurns=0,entries=0,deniedSpace=0,deniedYield=0;double calc=0,advanceMs=0,junctionMs=0;int maxSame=0;
 cudaEvent_t ev0,ev1,ev2;check(cudaEventCreate(&ev0));check(cudaEventCreate(&ev1));check(cudaEventCreate(&ev2));
 int steps=int(std::round(duration/dt));
 for(int step=0;step<steps;++step){float t=step*dt;
  auto start=std::chrono::steady_clock::now();check(cudaEventRecord(ev0));advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,t,dt,mode,exclusive);check(cudaGetLastError());check(cudaEventRecord(ev1));if(parallel){
   prepareJunctions<<<(std::max(n,512)+127)/128,128>>>(b,n,dm,dt,ds);check(cudaGetLastError());
   summarizeJunctions<<<(n+127)/128,128>>>(b,n,dr,dm,t,mode,ds);check(cudaGetLastError());
   requestJunctionEntry<<<(n+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq);check(cudaGetLastError());
   arbitrateJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq);check(cudaGetLastError());
   spawnJunctionVehicles<<<(nr*4+127)/128,128>>>(b,n,nr,ds);check(cudaGetLastError());
  }else{junctionStep<<<1,1>>>(b,n,dr,dm,dc,nm,t,dt,mode,exclusive);check(cudaGetLastError());}check(cudaEventRecord(ev2));check(cudaEventSynchronize(ev2));calc+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();float am,jm;check(cudaEventElapsedTime(&am,ev0,ev1));check(cudaEventElapsedTime(&jm,ev1,ev2));advanceMs+=am;junctionMs+=jm;std::swap(a,b);check(cudaMemcpy(cars.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));
  int active=0,stopped=0,inside=0,completed=0,pending=0;
  for(int i=0;i<n;++i){const auto& c=cars[i];active+=c.state==1||c.state==2;stopped+=(c.state==1||c.state==2)&&c.v<.5;inside+=c.state==2;completed+=c.state==3;pending+=c.state==0;
   invalid+=!std::isfinite(c.pos)||!std::isfinite(c.v)||c.v<0||c.pos<0||(c.state==1&&c.pos>roads[c.edges[c.k]].length+.001f)||(c.state==2&&c.pos>moves[c.mid].length+.001f);insideStops+=c.state==2&&c.v<.5f;
   deniedSpace+=c.denial==2;deniedYield+=c.denial==1||c.denial==3;
   if(c.event==1||c.event==2){auto m=moves[c.mid];entries++;redTurns+=c.event==2;events<<t+dt<<','<<i<<','<<c.event<<','<<c.mid<<','<<m.turn<<','<<m.axis<<','<<indication(m,t,mode)<<','<<c.entryWait<<'\n';}
   if((step+1)%2==0&&(c.state==1||c.state==2))traj<<t+dt<<','<<i<<','<<c.state<<','<<c.edges[c.k]<<','<<c.lanes[c.k]<<','<<c.mid<<','<<c.pos<<','<<c.v<<'\n';
  }
  std::vector<std::pair<int,float>> lanePositions;std::vector<int> occupiedMovements;
  for(const auto& c:cars){if(c.state==1)lanePositions.emplace_back(c.edges[c.k]*16+c.lanes[c.k],c.pos);if(c.state==2){occupiedMovements.push_back(c.mid);lanePositions.emplace_back(10000+c.mid,c.pos);}}
  std::sort(lanePositions.begin(),lanePositions.end());
  for(size_t i=1;i<lanePositions.size();++i)if(lanePositions[i].first==lanePositions[i-1].first&&lanePositions[i].second-lanePositions[i-1].second<SPACING-.001f)gapFailures++;
  for(size_t i=0;i<occupiedMovements.size();++i)for(size_t j=i+1;j<occupiedMovements.size();++j)conflictFailures+=conflicts[occupiedMovements[i]*nm+occupiedMovements[j]]&&(exclusive||occupiedMovements[i]!=occupiedMovements[j]);
  for(int m=0;m<nm;++m)maxSame=std::max(maxSame,int(std::count(occupiedMovements.begin(),occupiedMovements.end(),m)));
  // Check spacing across road/connector boundaries, not just within each container.
  for(const auto& c:cars)if(c.state==2){const auto& m=moves[c.mid];for(const auto& d:cars)if(d.state==1){
   if(d.edges[d.k]==m.out&&d.lanes[d.k]==m.outLane&&m.length-c.pos+d.pos<SPACING-.001f)gapFailures++;
   if(d.edges[d.k]==m.in&&d.lanes[d.k]==m.inLane&&roads[m.in].length-d.pos+c.pos<SPACING-.001f)gapFailures++;}}
  if((step+1)%2==0){series<<t+dt<<','<<active<<','<<stopped<<','<<inside<<','<<completed<<','<<pending<<'\n';for(int m=0;m<nm;++m)signals<<t<<','<<m<<','<<indication(moves[m],t,mode)<<'\n';}
 }
 int completed=0,active=0,pending=0;for(auto c:cars){completed+=c.state==3;active+=c.state==1||c.state==2;pending+=c.state==0;}
 std::ofstream metrics("metrics.json");metrics<<"{\"trips\":"<<n<<",\"completed\":"<<completed<<",\"active\":"<<active<<",\"pending\":"<<pending<<",\"steps\":"<<steps<<",\"calculation_seconds\":"<<calc<<",\"gap_failures\":"<<gapFailures<<",\"conflict_failures\":"<<conflictFailures<<",\"invalid_states\":"<<invalid<<",\"stopped_in_junction_samples\":"<<insideStops<<",\"entries\":"<<entries<<",\"right_on_red_entries\":"<<redTurns<<",\"downstream_blocked_decisions\":"<<deniedSpace<<",\"yield_blocked_decisions\":"<<deniedYield<<",\"parallel_junctions\":"<<parallel<<",\"exclusive\":"<<exclusive<<",\"advance_gpu_ms\":"<<advanceMs<<",\"junction_gpu_ms\":"<<junctionMs<<",\"max_same_movement_occupancy\":"<<maxSame<<"}\n";
 check(cudaEventDestroy(ev0));check(cudaEventDestroy(ev1));check(cudaEventDestroy(ev2));
 check(cudaFree(ds));check(cudaFree(dq));
 check(cudaFree(a));check(cudaFree(b));check(cudaFree(dr));check(cudaFree(dm));check(cudaFree(dc));
 std::cout<<"Junction run completed: "<<completed<<" / "<<n<<"; invariant failures "<<gapFailures+conflictFailures+invalid+insideStops<<std::endl;
 if(gapFailures||conflictFailures||invalid||insideStops)throw std::runtime_error("Junction invariant violation");
}
