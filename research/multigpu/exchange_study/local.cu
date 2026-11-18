// Local record domain; unchanged frozen physics. Sorted stable IDs preserve priority.
#define main frozen_correctness_main
#include "../source/research/multigpu/partitioned.cu"
#undef main
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <cub/cub.cuh>
#include "trace.cuh"
static int exchangeMode(){const char* s=getenv("LPSIM_EXCHANGE_MODE");int v=s?atoi(s):0;if(v<0||v>3)throw std::runtime_error("Invalid exchange mode");return v;}
struct Team {
 std::mutex m;std::condition_variable start,done;int generation=0,finished=0,k;bool stop=false;std::function<void(int)> fn;std::exception_ptr error;std::vector<std::thread> ts;
 Team(int count):k(count){for(int r=0;r<k;r++)ts.emplace_back([this,r]{int seen=0;for(;;){std::function<void(int)> f;{std::unique_lock<std::mutex> l(m);start.wait(l,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;f=fn;}try{f(r);}catch(...){std::lock_guard<std::mutex> l(m);error=std::current_exception();}{std::lock_guard<std::mutex> l(m);finished++;done.notify_one();}}});}
 void run(std::function<void(int)> f){{std::lock_guard<std::mutex> l(m);fn=f;finished=0;error=nullptr;generation++;}start.notify_all();std::unique_lock<std::mutex> l(m);done.wait(l,[&]{return finished==k;});if(error)std::rethrow_exception(error);}
 ~Team(){{std::lock_guard<std::mutex> l(m);stop=true;}start.notify_all();for(auto& t:ts)t.join();}
};
struct LocalMessage{int id;Car car;};
__global__ void packLocal(const Car* cars,const int* ids,const int* oldOwners,int n,int rank,const Road* roads,const Movement* moves,const int* jo,LocalMessage* retained,LocalMessage* send,int* count,int cap,int ranks){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||oldOwners[i]!=rank)return;Car c=cars[i];int own=authority(c,roads,moves,jo);int ghost=c.state==1?jo[roads[c.edges[c.k]].from]:own;
 if(own==rank){int s=atomicAdd(count+ranks,1);if(s<cap)retained[s]={ids[i],c};}
 else{int s=atomicAdd(count+own,1);if(s<cap)send[own*cap+s]={ids[i],c};atomicAdd(count+ranks+1,1);}
 if(ghost!=own){int s=atomicAdd(count+ghost,1);if(s<cap)send[ghost*cap+s]={ids[i],c};}
}
__global__ void localKeys(const LocalMessage* records,int n,int* keys,int* order){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){keys[i]=records[i].id;order[i]=i;}}
__global__ void gatherLocal(const LocalMessage* records,const int* keys,const int* order,int n,Car* cars,int* ids,int* owners,const Road* roads,const Movement* moves,const int* jo){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){Car c=records[order[i]].car;cars[i]=c;ids[i]=keys[i];owners[i]=authority(c,roads,moves,jo);}}
// Bit 0: persistent radix workspace. Bit 1: retain local slot references.
__global__ void packSlots(const Car* cars,const int* ids,const int* oldOwners,int n,int rank,const Road* roads,const Movement* moves,const int* jo,int* keys,int* order,LocalMessage* send,int* count,int cap,int ranks){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||oldOwners[i]!=rank)return;Car c=cars[i];int own=authority(c,roads,moves,jo);int ghost=c.state==1?jo[roads[c.edges[c.k]].from]:own;
 if(own==rank){int s=atomicAdd(count+ranks,1);if(s<cap){keys[s]=ids[i];order[s]=i;}}
 else{int s=atomicAdd(count+own,1);if(s<cap)send[own*cap+s]={ids[i],c};atomicAdd(count+ranks+1,1);}
 if(ghost!=own){int s=atomicAdd(count+ghost,1);if(s<cap)send[ghost*cap+s]={ids[i],c};}
}
__global__ void incomingKeys(const LocalMessage* records,int incoming,int offset,int* keys,int* order){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<incoming){keys[offset+i]=records[i].id;order[offset+i]=~i;}}
__global__ void gatherSlots(const Car* old,const LocalMessage* incoming,const int* keys,const int* order,int n,Car* cars,int* ids,int* owners,const Road* roads,const Movement* moves,const int* jo){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int slot=order[i];Car c=slot>=0?old[slot]:incoming[~slot].car;cars[i]=c;ids[i]=keys[i];owners[i]=authority(c,roads,moves,jo);}}
struct Local {
 std::unique_ptr<Partition> p;int total,capacity,count,rank,ranks;int mode;int *sortedKeys=nullptr,*sortedOrder=nullptr;void* sortTemp=nullptr;size_t sortBytes=0;ExchangeTrace trace;int *ids,*keys,*order,*counts,*host;LocalMessage *retained,*send;ReservationMessage *rs,*rr;int *rkeys;std::vector<int> rn,ro;std::vector<int> observedIDs;
 Local(int r,int gpu,int k,const Model& model,const std::vector<Car>& initial,const std::vector<int>& jo):total(initial.size()),capacity(total),rank(r),ranks(k),mode(exchangeMode()),trace(r),rn(k),ro(k){
  std::vector<Car> visible;std::vector<int> own,gids;
  for(int i=0;i<total;i++){int o=authority(initial[i],model.roads.data(),model.moves.data(),jo.data());if(o==r||(initial[i].state==1&&jo[model.roads[initial[i].edges[initial[i].k]].from]==r)){visible.push_back(initial[i]);own.push_back(o);gids.push_back(i);}}
  count=visible.size();capacity=std::min(total,std::max(256,(count*3+1)/2));visible.resize(capacity);own.resize(capacity,r);gids.resize(capacity,-1);for(int i=count;i<capacity;i++)visible[i].state=-1;
  p.reset(new Partition(r,gpu,model,visible,own,jo,false,1));ids=device(gids);keys=device(std::vector<int>(capacity));order=device(std::vector<int>(capacity));counts=device(std::vector<int>(k+2));check(cudaMallocHost(&host,(k+2)*sizeof(int)));check(cudaMalloc(&retained,capacity*sizeof(LocalMessage)));check(cudaMalloc(&send,k*capacity*sizeof(LocalMessage)));
  std::vector<int> ks;for(int d=0;d<k;d++){ro[d]=ks.size();for(int e=0;e<model.nr;e++)if(jo[model.roads[e].from]==r&&jo[model.roads[e].to]==d&&d!=r)for(int l=0;l<model.roads[e].lanes;l++){ks.push_back(e*8+l);rn[d]++;}}
  if(mode&1){sortedKeys=device(std::vector<int>(capacity));sortedOrder=device(std::vector<int>(capacity));check(cub::DeviceRadixSort::SortPairs(nullptr,sortBytes,keys,sortedKeys,order,sortedOrder,capacity));check(cudaMalloc(&sortTemp,sortBytes));}
  rkeys=device(ks.empty()?std::vector<int>{0}:ks);check(cudaMalloc(&rs,std::max(1,int(ks.size()))*sizeof(ReservationMessage)));check(cudaMalloc(&rr,k*model.nr*8*sizeof(ReservationMessage)));resize();
 }
 void resize(){p->n=count;p->hot.count=count;p->bulk->n=count;p->context->n=count;p->requestIndex->n=count;}
 void motion(float t){p->select();if(count){check(cudaMemcpyAsync(p->b,p->a,count*sizeof(Car),cudaMemcpyDeviceToDevice));p->motion(t,.25f,0);}else prepareJunctions<<<(p->cap+127)/128,128>>>(p->b,0,p->moves,0,p->ds);}
 void junction(float t){p->select();if(count)p->junction(t,0);}
 void inspect(){p->select();p->inspection.resize(count);observedIDs.resize(count);if(count){check(cudaMemcpy(p->inspection.data(),p->b,count*sizeof(Car),cudaMemcpyDeviceToHost));check(cudaMemcpy(observedIDs.data(),ids,count*sizeof(int),cudaMemcpyDeviceToHost));}}
};
struct Exchange {
 std::vector<Local*>& ps;Team& team;int ranks;unsigned long long bytes=0,migrations=0;Exchange(std::vector<Local*>& p,Team& t):ps(p),team(t),ranks(p.size()){
 for(auto* dst:ps){dst->p->select();for(auto* src:ps)if(src->p->gpu!=dst->p->gpu){int access;check(cudaDeviceCanAccessPeer(&access,dst->p->gpu,src->p->gpu));if(!access)throw std::runtime_error("P2P required");auto e=cudaDeviceEnablePeerAccess(src->p->gpu,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else check(e);}}}
 void copy(void* dst,int d,const void* src,int s,size_t n){if(!n)return;if(ps[d]->p->gpu==ps[s]->p->gpu)check(cudaMemcpyAsync(dst,src,n,cudaMemcpyDeviceToDevice));else check(cudaMemcpyPeerAsync(dst,ps[d]->p->gpu,src,ps[s]->p->gpu,n));}
 void state(int phase){
 team.run([&](int r){auto* a=ps[r];a->p->select();a->trace.reset(phase?"exchange1.pack":"exchange0.pack");
 a->trace.part("pack",[&]{check(cudaMemsetAsync(a->counts,0,(ranks+2)*4));if(a->count){if(a->mode&2)packSlots<<<(a->count+127)/128,128>>>(a->p->b,a->ids,a->p->owners,a->count,r,a->p->roads,a->p->moves,a->p->jo,a->keys,a->order,a->send,a->counts,a->capacity,ranks);else packLocal<<<(a->count+127)/128,128>>>(a->p->b,a->ids,a->p->owners,a->count,r,a->p->roads,a->p->moves,a->p->jo,a->retained,a->send,a->counts,a->capacity,ranks);}});
 a->trace.part("count_d2h",[&]{check(cudaMemcpyAsync(a->host,a->counts,(ranks+2)*4,cudaMemcpyDeviceToHost));});a->trace.sync();});
 for(int s=0;s<ranks;s++){auto* a=ps[s];for(int d=0;d<ranks;d++){if(a->host[d]>a->capacity)throw std::runtime_error("Send overflow");if(d!=s)bytes+=a->host[d]*sizeof(LocalMessage);}if(phase&&a->host[ranks+1])throw std::runtime_error("Wrong migration phase");migrations+=a->host[ranks+1];}
 team.run([&](int d){auto* a=ps[d];a->p->select();a->trace.reset(phase?"exchange1.transfer":"exchange0.transfer");int kept=a->host[ranks],size=kept;
 a->trace.part("peer_copies",[&]{for(int s=0;s<ranks;s++){int n=ps[s]->host[d];if(size+n>a->capacity)throw std::runtime_error("Receive overflow");copy(a->retained+size-((a->mode&2)?kept:0),d,ps[s]->send+d*ps[s]->capacity,s,n*sizeof(LocalMessage));size+=n;}});a->count=size;a->trace.sync();});
 team.run([&](int r){auto* a=ps[r];a->p->select();a->trace.reset(phase?"exchange1.list":"exchange0.list");a->resize();int n=a->count;
 if(n){a->trace.part("keys",[&]{if(a->mode&2){int kept=a->host[ranks],incoming=n-kept;if(incoming)incomingKeys<<<(incoming+127)/128,128>>>(a->retained,incoming,kept,a->keys,a->order);}else localKeys<<<(n+127)/128,128>>>(a->retained,n,a->keys,a->order);});
 int* keys=a->keys;int* order=a->order;
 a->trace.part("sort",[&]{if(a->mode&1){check(cub::DeviceRadixSort::SortPairs(a->sortTemp,a->sortBytes,a->keys,a->sortedKeys,a->order,a->sortedOrder,n));keys=a->sortedKeys;order=a->sortedOrder;}else thrust::sort_by_key(thrust::device_pointer_cast(a->keys),thrust::device_pointer_cast(a->keys)+n,thrust::device_pointer_cast(a->order));});
 a->trace.part("gather",[&]{if(a->mode&2){gatherSlots<<<(n+127)/128,128>>>(a->p->b,a->retained,keys,order,n,a->p->a,a->ids,a->p->owners,a->p->roads,a->p->moves,a->p->jo);std::swap(a->p->a,a->p->b);}else gatherLocal<<<(n+127)/128,128>>>(a->retained,keys,order,n,a->p->b,a->ids,a->p->owners,a->p->roads,a->p->moves,a->p->jo);});}
 if(phase==0)a->trace.part("junction_refresh",[&]{prepareJunctions<<<(std::max(n,a->p->cap)+127)/128,128>>>(a->p->b,n,a->p->moves,0,a->p->ds);});a->trace.sync();});
 }
 void reservations(int epoch){team.run([&](int r){auto* a=ps[r];a->p->select();a->trace.reset("reservation.pack");int n=0;for(int c:a->rn)n+=c;a->trace.part("pack",[&]{if(n)packReservations<<<(n+127)/128,128>>>(a->p->ds,a->rkeys,a->p->roads,a->p->jo,n,r,epoch,a->rs);});a->trace.sync();});for(auto* a:ps)for(int c:a->rn)bytes+=c*sizeof(ReservationMessage);
 team.run([&](int d){auto* a=ps[d];a->p->select();a->trace.reset("reservation.transfer");a->trace.part("peer_copies",[&]{for(int s=0;s<ranks;s++){int n=ps[s]->rn[d];if(n)copy(a->rr+s*a->p->nr*8,d,ps[s]->rs+ps[s]->ro[d],s,n*sizeof(ReservationMessage));}});a->trace.part("unpack",[&]{for(int s=0;s<ranks;s++){int n=ps[s]->rn[d];if(n)unpackReservations<<<(n+127)/128,128>>>(a->p->ds,a->rr+s*a->p->nr*8,n);}});a->trace.sync();});}
};
static double elapsed(std::chrono::steady_clock::time_point a){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-a).count();}
int main(int argc,char** argv){try{
 if(argc!=9)throw std::runtime_error("local MODEL SNAPSHOT OWNERS STEPS OUT RANKS VERIFY DIAGNOSTIC");setenv("LPSIM_MOTION_CONTEXT","3",1);Model model(argv[1]);std::ifstream f(argv[2],std::ios::binary);int n;float start;f.read((char*)&n,4);f.read((char*)&start,4);if(n<1)throw std::runtime_error("Invalid n");std::vector<Car> initial(n);f.read((char*)initial.data(),n*sizeof(Car));if(!f)throw std::runtime_error("Invalid snapshot");int ranks=atoi(argv[6]),steps=atoi(argv[4]);bool verify=atoi(argv[7]),diagnostic=atoi(argv[8]);if(ranks<1||ranks>3||steps<1)throw std::runtime_error("Invalid ranks/steps");std::ifstream map(argv[3]);std::vector<int> jo(model.nj),owners(n);for(int& j:jo)if(!(map>>j)||j<0||j>=ranks)throw std::runtime_error("Invalid map");for(int i=0;i<n;i++)owners[i]=authority(initial[i],model.roads.data(),model.moves.data(),jo.data());std::string out=argv[5];auto setup=std::chrono::steady_clock::now();Team team(ranks);std::vector<Local*> parts;for(int r=0;r<ranks;r++)parts.push_back(new Local(r,getenv("LPSIM_LOGICAL")?0:r,ranks,model,initial,jo));Exchange exchange(parts,team);std::unique_ptr<Partition> reference;if(verify)reference.reset(new Partition(-1,0,model,initial,owners,jo,false,1));double setupMs=elapsed(setup),total=0;std::vector<Car> observed(n);std::ofstream times(out+"/steps.csv"),pop(out+"/populations.csv");times<<"step,total_ms,motion_ms,exchange0_ms,junction_ms,reservation_ms,spawn_ms,exchange1_ms\n";pop<<"step,rank,owned,pending,stopped,moving,connector,completed,ghosts,processed_records,reserved_capacity\n";
 for(int step=0;step<steps;step++){float t=start+step*.25f;auto begin=std::chrono::steady_clock::now(),z=begin;auto stage=[&](const char* label,auto fn){team.run([&](int r){auto* a=parts[r];a->p->select();a->trace.step=step;a->trace.reset(label);a->trace.part(label,[&]{fn(a);});a->trace.sync();});double ms=elapsed(z);z=std::chrono::steady_clock::now();return ms;};double motion=stage("motion",[&](Local* a){a->motion(t);});if(ranks>1)exchange.state(0);double ex0=elapsed(z);z=std::chrono::steady_clock::now();double junction=stage("junction",[&](Local* a){a->junction(t);});if(ranks>1)exchange.reservations(step);double res=elapsed(z);z=std::chrono::steady_clock::now();double spawn=stage("spawn",[&](Local* a){a->p->spawn();});if(ranks>1)exchange.state(1);double ex1=elapsed(z),ms=elapsed(begin);total+=ms;times<<step<<','<<ms<<','<<motion<<','<<ex0<<','<<junction<<','<<res<<','<<spawn<<','<<ex1<<'\n';
 if(verify||diagnostic||step==steps-1){std::vector<int> seen(n);for(auto* a:parts){a->inspect();int counts[7]={0};for(int k=0;k<a->count;k++){int id=a->observedIDs[k];auto c=a->p->inspection[k];if(id<0||id>=n||(k&&id<=a->observedIDs[k-1]))throw std::runtime_error("Duplicate/unordered stable ID");int own=authority(c,model.roads.data(),model.moves.data(),jo.data());if(own==a->rank){observed[id]=c;seen[id]++;counts[0]++;counts[c.state==0?1:c.state==1?(c.v<.5f?2:3):c.state==2?4:5]++;}else counts[6]++;}pop<<step<<','<<a->rank;for(int v:counts)pop<<','<<v;pop<<','<<a->count<<','<<a->capacity<<'\n';}for(int c:seen)if(c!=1)throw std::runtime_error("Missing/duplicate authority");
 if(verify){reference->motion(t,.25f,0);reference->junction(t,0);reference->spawn();reference->inspect();for(int i=0;i<n;i++)if(memcmp(&observed[i],&reference->inspection[i],sizeof(Car))){std::cerr<<"step="<<step<<" id="<<i<<" state="<<observed[i].state<<" pos="<<observed[i].pos<<" expected="<<reference->inspection[i].pos<<std::endl;throw std::runtime_error("Exact mismatch");}for(auto* a:parts)for(int k=0;k<a->count;k++)if(memcmp(&a->p->inspection[k],&observed[a->observedIDs[k]],sizeof(Car)))throw std::runtime_error("Ghost mismatch");validatePhysical(observed,model);std::swap(reference->a,reference->b);}}
 for(auto* a:parts)std::swap(a->p->a,a->p->b);
 }
 std::ofstream traceFile(out+"/trace.csv");traceFile<<"step,rank,stage,component,gpu_ms,host_begin_ns,host_end_ns\n";for(auto* a:parts)a->trace.write(traceFile);
 std::ofstream final(out+"/final.bin",std::ios::binary);float end=start+steps*.25f;final.write((char*)&n,4);final.write((char*)&end,4);final.write((char*)observed.data(),n*sizeof(Car));std::ofstream metrics(out+"/metrics.json");metrics<<"{\"ranks\":"<<ranks<<",\"steps\":"<<steps<<",\"requested\":"<<n<<",\"compute_ms\":"<<total<<",\"setup_ms\":"<<setupMs<<",\"wire_bytes\":"<<exchange.bytes<<",\"migrations\":"<<exchange.migrations<<",\"verified\":"<<(verify?"true":"false")<<"}\n";std::cout<<"PASS local ranks="<<ranks<<" steps="<<steps<<" ms="<<total<<std::endl;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
