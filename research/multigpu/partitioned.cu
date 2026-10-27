// Correctness prototype: explicit host-staged transport, independent device buffers.
#define LPSIM_SCALABLE_LIBRARY
#include "../../src/simulator/scalable_junction_simulator.cu"
#include "../../tools/incremental_leaders.cuh"
#include <memory>
#include <set>
#include <cstring>

struct StatePayload { int state,k,mid,event,denial; float pos,v,stopped,entryWait; };
struct StateMessage { int id,src,dst,newOwner,epoch,phase; StatePayload value; };
struct ReservationMessage { int key,src,dst,epoch,reserved,owner; };
static_assert(sizeof(Car)==428,"Snapshot ABI changed");
static_assert(sizeof(StateMessage)==60,"Wire format changed");
__host__ __device__ StatePayload payload(const Car& c){return {c.state,c.k,c.mid,c.event,c.denial,c.pos,c.v,c.stopped,c.entryWait};}
__host__ __device__ void apply(Car& c,const StatePayload& p){c.state=p.state;c.k=p.k;c.mid=p.mid;c.event=p.event;c.denial=p.denial;c.pos=p.pos;c.v=p.v;c.stopped=p.stopped;c.entryWait=p.entryWait;}
__host__ __device__ int authority(const Car& c,const Road* r,const Movement* m,const int* jo){return c.state==2?jo[m[c.mid].junction]:jo[r[c.edges[c.k]].to];}
__global__ void packState(const Car* c,int n,const Road* r,const Movement* m,const int* jo,const int* owners,int rank,int epoch,int phase,StateMessage* out,int* count,int cap){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n||owners[i]!=rank)return;
 int owner=authority(c[i],r,m,jo),ghost=c[i].state==1?jo[r[c[i].edges[c[i].k]].from]:owner;
 if(owner!=rank){int k=atomicAdd(count,1);if(k<cap)out[k]={i,rank,owner,owner,epoch,phase,payload(c[i])};}
 if(ghost!=owner){int k=atomicAdd(count,1);if(k<cap)out[k]={i,rank,ghost,owner,epoch,phase,payload(c[i])};}
}
__global__ void clearGhosts(Car* c,int n,const int* owners,int rank){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&owners[i]!=rank)c[i].state=-1;}
__global__ void unpackState(Car* c,const StateMessage* messages,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)apply(c[messages[i].id],messages[i].value);}
__global__ void packReservations(const JunctionSummary* s,const int* keys,const Road* roads,const int* jo,int n,int rank,int epoch,ReservationMessage* out){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){int key=keys[i];out[i]={key,rank,jo[roads[key/8].to],epoch,s->reserved[key],s->owner[key]};}
}
__global__ void unpackReservations(JunctionSummary* s,const ReservationMessage* messages,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){auto m=messages[i];s->reserved[m.key]=m.reserved;s->owner[m.key]=m.owner;}}
// Do not generate requests for replicated receiving-road ghosts.
__global__ void filterRequests(EntryRequest* q,int n,const int* owners,int rank,int nm){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&owners[i]!=rank)q[i]={-1,-1,nm,0};}

struct Model {
 int nr,nm,nj; std::vector<Road> roads;std::vector<Movement> moves;std::vector<int> csr,offsets;
 explicit Model(const char* path){std::ifstream f(path);if(!(f>>nr>>nm>>nj)||nr<1||nm<1||nj<1)throw std::runtime_error("Invalid model header");roads.resize(nr);moves.resize(nm);
 for(auto& r:roads)if(!(f>>r.from>>r.to>>r.lanes>>r.length>>r.speed)||r.from<0||r.from>=nj||r.to<0||r.to>=nj||r.lanes<1||r.lanes>8||r.length<=0)throw std::runtime_error("Invalid road");
 offsets.resize(nj+1);for(auto& m:moves){if(!(f>>m.junction>>m.in>>m.out>>m.inLane>>m.outLane>>m.turn>>m.axis>>m.length>>m.rtor)||m.in<0||m.in>=nr||m.out<0||m.out>=nr||m.junction<0||m.junction>=nj||roads[m.in].to!=m.junction||roads[m.out].from!=m.junction||m.inLane<0||m.inLane>=roads[m.in].lanes||m.outLane<0||m.outLane>=roads[m.out].lanes||m.length<=0)throw std::runtime_error("Invalid movement ownership");offsets[m.junction+1]++;}
 for(int j=0;j<nj;++j)offsets[j+1]+=offsets[j];std::string format;int entries;f>>format>>entries;if(format!="SPARSE"||entries<0)throw std::runtime_error("Sparse model required");std::vector<std::vector<int>> rows(nm);
 for(int k=0;k<entries;++k){int a,b;if(!(f>>a>>b)||a<0||a>=nm||b<0||b>=nm||moves[a].junction!=moves[b].junction)throw std::runtime_error("Invalid conflict dependency");rows[a].push_back(b);}csr.resize(nm+1);
 for(int i=0;i<nm;++i){std::sort(rows[i].begin(),rows[i].end());rows[i].erase(std::unique(rows[i].begin(),rows[i].end()),rows[i].end());csr[i+1]=csr[i]+rows[i].size();}for(auto& row:rows)csr.insert(csr.end(),row.begin(),row.end());
 }
};
struct Partition {
 int rank,gpu,n,nr,nm,nj,cap,messageCap,junctionCount,reservationCount;
 Car *a,*b;Road* roads;Movement* moves;int *csr,*jo,*owners,*junctions,*reservationKeys,*messageCount;
 JunctionSummary summary{},*ds;EntryRequest* requests;CompactState hot;
 std::unique_ptr<LeaderIndex> bulk;std::unique_ptr<IncrementalLeaders> incremental;
 std::unique_ptr<MotionContext> context;std::unique_ptr<JunctionRequestIndex> requestIndex;
 StateMessage *wire,*hostSend,*hostReceive;ReservationMessage *resWire,*resSend,*resReceive;
 std::vector<Car> inspection; // Verification only, never used to advance a partition.
 void select(){check(cudaSetDevice(gpu));}
 Partition(int rank_,int gpu_,const Model& model,const std::vector<Car>& initial,const std::vector<int>& ownership,const std::vector<int>& junctionOwners,bool useIncremental,int capacity):rank(rank_),gpu(gpu_),n(initial.size()),nr(model.nr),nm(model.nm),nj(model.nj),messageCap(capacity),inspection(n){
 select();roads=device(model.roads);moves=device(model.moves);csr=device(model.csr);jo=device(junctionOwners);owners=device(ownership);
 auto visible=initial;for(int i=0;i<n;++i)if(rank>=0&&ownership[i]!=rank&&!(initial[i].state==1&&junctionOwners[model.roads[initial[i].edges[initial[i].k]].from]==rank))visible[i].state=-1;
 a=device(visible);b=device(visible);hot={device(std::vector<float>(14*n)),n};
 if(useIncremental)incremental.reset(new IncrementalLeaders(n,nr,false));else bulk.reset(new LeaderIndex(n,true,true));
 context.reset(new MotionContext(nr,nm,n));requestIndex.reset(new JunctionRequestIndex(n,nj,true));
 cap=summary.capacity=std::max(nr*8,nm);summary.nearest=device(std::vector<float>(cap));summary.arrival=device(std::vector<float>(cap));summary.tail=device(std::vector<float>(cap));summary.busy=device(std::vector<int>(cap));summary.reserved=device(std::vector<int>(cap));summary.owner=device(std::vector<int>(cap));summary.spawn=device(std::vector<int>(cap));summary.granted=device(std::vector<int>(nm));summary.junctionOffsets=device(model.offsets);ds=device(std::vector<JunctionSummary>{summary});requests=device(std::vector<EntryRequest>(n));
 std::vector<int> ownedJunctions,keys;for(int j=0;j<nj;++j)if(rank<0||junctionOwners[j]==rank)ownedJunctions.push_back(j);
 for(int r=0;r<nr;++r)if(rank>=0&&junctionOwners[model.roads[r].from]==rank&&junctionOwners[model.roads[r].to]!=rank)for(int l=0;l<model.roads[r].lanes;++l)keys.push_back(r*8+l);
 junctionCount=ownedJunctions.size();reservationCount=keys.size();junctions=device(ownedJunctions.empty()?std::vector<int>{0}:ownedJunctions);reservationKeys=device(keys.empty()?std::vector<int>{0}:keys);
 wire=device(std::vector<StateMessage>(messageCap));messageCount=device(std::vector<int>(1));check(cudaMallocHost(&hostSend,messageCap*sizeof(StateMessage)));check(cudaMallocHost(&hostReceive,messageCap*sizeof(StateMessage)));
 resWire=device(std::vector<ReservationMessage>(nr*8));check(cudaMallocHost(&resSend,nr*8*sizeof(ReservationMessage)));check(cudaMallocHost(&resReceive,nr*8*sizeof(ReservationMessage)));
 }
 void motion(float t,float dt,int mode){select();int* leaders;if(incremental){incremental->build(a);leaders=incremental->bulk.leaders;}else{bulk->build(a);leaders=bulk->leaders;}
 context->build(a,moves,csr,t,mode,false);extractCompact<<<(n+127)/128,128>>>(a,hot);advanceCompact<<<(n+127)/128,128>>>(hot,b,n,roads,moves,csr,nm,t,dt,mode,false,leaders,rank<0?nullptr:owners,rank,context->head,context->links,context->available);
 prepareJunctions<<<(std::max(n,cap)+127)/128,128>>>(b,n,moves,dt,ds);check(cudaGetLastError());}
 void junction(float t,int mode){select();summarizeJunctions<<<(n+127)/128,128>>>(b,n,roads,moves,t,mode,ds);requestJunctionEntry<<<(n+127)/128,128>>>(b,n,roads,moves,csr,nm,t,mode,false,ds,requests);if(rank>=0)filterRequests<<<(n+127)/128,128>>>(requests,n,owners,rank,nm);requestIndex->build(requests,moves);
 if(junctionCount)arbitrateIndexedJunctions<<<(junctionCount+127)/128,128>>>(b,n,roads,moves,csr,nm,nj,false,ds,requests,requestIndex->keys,requestIndex->begin,requestIndex->end,junctions,junctionCount);check(cudaGetLastError());}
 void spawn(){select();spawnJunctionVehicles<<<(nr*8+127)/128,128>>>(b,n,nr,ds);check(cudaGetLastError());}
 void inspect(){select();check(cudaMemcpy(inspection.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));}
};
struct Transport {
 const Model& model;const std::vector<int>& jo;std::vector<int>& owners;std::vector<Partition*>& partitions;
 unsigned long long messages=0,bytes=0,migrations=0,reservations=0,controlBytes=0;int peak=0;
 Transport(const Model& m,const std::vector<int>& j,std::vector<int>& o,std::vector<Partition*>& p):model(m),jo(j),owners(o),partitions(p){}
 void exchange(int epoch,int phase){
 std::vector<int> old=owners;int counts[3]={0,0,0};std::set<std::pair<int,int>> seen;std::set<int> moved;
 for(auto* p:partitions){p->select();check(cudaMemset(p->messageCount,0,4));packState<<<(p->n+127)/128,128>>>(p->b,p->n,p->roads,p->moves,p->jo,p->owners,p->rank,epoch,phase,p->wire,p->messageCount,p->messageCap);int count;check(cudaMemcpy(&count,p->messageCount,4,cudaMemcpyDeviceToHost));if(count>p->messageCap)throw std::runtime_error("State outbox capacity exceeded; no messages applied");check(cudaMemcpy(p->hostSend,p->wire,count*sizeof(StateMessage),cudaMemcpyDeviceToHost));
 std::sort(p->hostSend,p->hostSend+count,[](const StateMessage& a,const StateMessage& b){return std::tie(a.dst,a.id)<std::tie(b.dst,b.id);});peak=std::max(peak,count);
 for(int k=0;k<count;++k){auto msg=p->hostSend[k];if(msg.epoch!=epoch||msg.phase!=phase||msg.src!=p->rank||msg.dst<0||msg.dst>=3||msg.id<0||msg.id>=int(owners.size())||old[msg.id]!=msg.src||msg.newOwner<0||msg.newOwner>=3||!seen.emplace(msg.dst,msg.id).second)throw std::runtime_error("Invalid/duplicate state message");
 if(msg.newOwner!=msg.src){if(phase!=0)throw std::runtime_error("Ownership changed outside connector-exit phase");owners[msg.id]=msg.newOwner;moved.insert(msg.id);}
 auto* dest=partitions[msg.dst];if(counts[msg.dst]>=dest->messageCap)throw std::runtime_error("State inbox capacity exceeded; no messages applied");dest->hostReceive[counts[msg.dst]++]=msg;
 }}
 migrations+=moved.size();controlBytes+=moved.size()*sizeof(int)*4*3;
 for(auto* p:partitions){p->select();check(cudaMemcpy(p->owners,owners.data(),owners.size()*4,cudaMemcpyHostToDevice));clearGhosts<<<(p->n+127)/128,128>>>(p->b,p->n,p->owners,p->rank);
 int count=counts[p->rank];if(count){check(cudaMemcpy(p->wire,p->hostReceive,count*sizeof(StateMessage),cudaMemcpyHostToDevice));unpackState<<<(count+127)/128,128>>>(p->b,p->wire,count);}messages+=count;bytes+=count*sizeof(StateMessage);peak=std::max(peak,count);check(cudaGetLastError());}
 }
 void reservationExchange(int epoch){int counts[3]={0,0,0};std::set<int> seen;
 for(auto* p:partitions){p->select();int count=p->reservationCount;if(count){packReservations<<<(count+127)/128,128>>>(p->ds,p->reservationKeys,p->roads,p->jo,count,p->rank,epoch,p->resWire);check(cudaMemcpy(p->resSend,p->resWire,count*sizeof(ReservationMessage),cudaMemcpyDeviceToHost));}
 for(int k=0;k<count;++k){auto m=p->resSend[k];if(m.epoch!=epoch||m.src!=p->rank||m.key<0||m.key>=model.nr*8||jo[model.roads[m.key/8].from]!=m.src||jo[model.roads[m.key/8].to]!=m.dst||m.src==m.dst||!seen.insert(m.key).second)throw std::runtime_error("Invalid reservation authority");if(counts[m.dst]>=model.nr*8)throw std::runtime_error("Reservation inbox overflow");partitions[m.dst]->resReceive[counts[m.dst]++]=m;}}
 for(auto* p:partitions){p->select();int count=counts[p->rank];if(count){check(cudaMemcpy(p->resWire,p->resReceive,count*sizeof(ReservationMessage),cudaMemcpyHostToDevice));unpackReservations<<<(count+127)/128,128>>>(p->ds,p->resWire,count);}reservations+=count;bytes+=count*sizeof(ReservationMessage);}
 }
};

// Invariants cover road, connector and cross-boundary following gaps and conflicts.
void validatePhysical(const std::vector<Car>& cars,const Model& model){
 std::vector<std::vector<float>> lanes(model.nr*8),connectors(model.nm);std::vector<int> busy;
 for(const auto& c:cars){if(c.state<0||c.state>3||c.k<0||c.k>=c.n||!std::isfinite(c.pos)||!std::isfinite(c.v)||c.pos<0||c.v<0)throw std::runtime_error("Invalid physical state");if(c.state==1){if(c.pos>model.roads[c.edges[c.k]].length+.001f)throw std::runtime_error("Road overrun");lanes[c.edges[c.k]*8+c.lanes[c.k]].push_back(c.pos);}if(c.state==2){if(c.mid<0||c.mid>=model.nm||c.v<.5f||c.pos>model.moves[c.mid].length+.001f)throw std::runtime_error("Invalid connector");connectors[c.mid].push_back(c.pos);}}
 auto gaps=[](std::vector<float>& xs){std::sort(xs.begin(),xs.end());for(size_t i=1;i<xs.size();++i)if(xs[i]-xs[i-1]<SPACING-.001f)throw std::runtime_error("Following gap violation");};for(auto& xs:lanes)gaps(xs);for(int i=0;i<model.nm;++i)if(!connectors[i].empty()){gaps(connectors[i]);busy.push_back(i);auto m=model.moves[i];auto& out=lanes[m.out*8+m.outLane];auto& in=lanes[m.in*8+m.inLane];if(!out.empty()&&m.length-connectors[i].back()+out.front()<SPACING-.001f)throw std::runtime_error("Receiving gap violation");if(!in.empty()&&model.roads[m.in].length-in.back()+connectors[i].front()<SPACING-.001f)throw std::runtime_error("Incoming gap violation");}
 for(int i:busy)for(int k=model.csr[i];k<model.csr[i+1];++k){int j=model.csr[model.nm+1+k];if(i!=j&&!connectors[j].empty())throw std::runtime_error("Conflicting movements occupied");}
}
int main(int argc,char** argv){try{
 if(argc<7)throw std::runtime_error("Usage: partitioned MODEL SNAPSHOT OWNERS STEPS OUTPUT_DIR bulk|incremental [mode=0] [devices=0,0,0] [message-cap=2N]");
 setenv("LPSIM_MOTION_CONTEXT","3",1);Model model(argv[1]);std::ifstream s(argv[2],std::ios::binary);int n;float start;s.read((char*)&n,4);s.read((char*)&start,4);if(!s||n<=0)throw std::runtime_error("Invalid snapshot header");std::vector<Car> cars(n);s.read((char*)cars.data(),n*sizeof(Car));if(!s)throw std::runtime_error("Invalid snapshot payload");
 for(auto c:cars){if(c.n<2||c.n>MAX_ROUTE||c.k<0||c.k>=c.n||c.state<0||c.state>3)throw std::runtime_error("Invalid snapshot route");for(int k=0;k<c.n;++k)if(c.edges[k]<0||c.edges[k]>=model.nr||c.lanes[k]<0||c.lanes[k]>=model.roads[c.edges[k]].lanes)throw std::runtime_error("Invalid snapshot lane");for(int k=0;k+1<c.n;++k){int mid=c.moves[k];if(mid<0||mid>=model.nm||model.moves[mid].in!=c.edges[k]||model.moves[mid].out!=c.edges[k+1]||model.moves[mid].inLane!=c.lanes[k]||model.moves[mid].outLane!=c.lanes[k+1])throw std::runtime_error("Invalid snapshot movement");}}
 std::ifstream map(argv[3]);std::vector<int> jo(model.nj);for(int& j:jo)if(!(map>>j)||j<0||j>=3)throw std::runtime_error("Invalid junction partition");int extra;if(map>>extra)throw std::runtime_error("Excess partition entries");
 int steps=std::stoi(argv[4]);if(steps<1)throw std::runtime_error("Positive step count required");std::string out=argv[5],option=argv[6];if(option!="bulk"&&option!="incremental")throw std::runtime_error("Index must be bulk or incremental");bool inc=option=="incremental";int mode=argc>7?std::stoi(argv[7]):0;if(mode<0||mode>2)throw std::runtime_error("Mode must be 0..2");int devices[3]={0,0,0};if(argc>8&&sscanf(argv[8],"%d,%d,%d",devices,devices+1,devices+2)!=3)throw std::runtime_error("Three device IDs required");int messageCap=argc>9?std::stoi(argv[9]):2*n;if(messageCap<1)throw std::runtime_error("Positive message capacity required");
 std::vector<int> owners(n);for(int i=0;i<n;++i)owners[i]=authority(cars[i],model.roads.data(),model.moves.data(),jo.data());
 Partition reference(-1,devices[0],model,cars,owners,jo,false,messageCap);std::vector<Partition*> partitions;for(int p=0;p<3;++p)partitions.push_back(new Partition(p,devices[p],model,cars,owners,jo,inc,messageCap));Transport transport(model,jo,owners,partitions);
 std::ofstream series(out+"/steps.csv");if(!series)throw std::runtime_error("Output directory must exist");series<<"step,time,pending,road_stopped,road_moving,connector,completed,state_messages,reservation_messages,migrations,wire_bytes,control_bytes,owner_map_host_to_device_bytes,partition0_active,partition1_active,partition2_active,reference_wall_ms,partition_wall_ms,verification_ms\n";std::vector<Car> observed(n);auto begin=std::chrono::steady_clock::now();double referenceMs=0,partitionMs=0,verifyMs=0;int pending=0,stopped=0,moving=0,connector=0,completed=0;
 for(int step=0;step<steps;++step){float t=start+step*.25f;auto z=std::chrono::steady_clock::now();reference.motion(t,.25f,mode);reference.junction(t,mode);reference.spawn();check(cudaDeviceSynchronize());auto r=std::chrono::steady_clock::now();referenceMs+=std::chrono::duration<double,std::milli>(r-z).count();
 auto oldMessages=transport.messages,oldReservations=transport.reservations,oldMigrations=transport.migrations,oldBytes=transport.bytes,oldControl=transport.controlBytes;
 for(auto* p:partitions)p->motion(t,.25f,mode);transport.exchange(step,0);for(auto* p:partitions)p->junction(t,mode);transport.reservationExchange(step);for(auto* p:partitions)p->spawn();transport.exchange(step,1);for(auto* p:partitions){p->select();check(cudaDeviceSynchronize());}auto done=std::chrono::steady_clock::now();partitionMs+=std::chrono::duration<double,std::milli>(done-r).count();
 reference.inspect();for(auto* p:partitions)p->inspect();int active[3]={0,0,0};pending=stopped=moving=connector=completed=0;
 for(int i=0;i<n;++i){auto c=partitions[owners[i]]->inspection[i];observed[i]=c;if(authority(c,model.roads.data(),model.moves.data(),jo.data())!=owners[i])throw std::runtime_error("Owner mismatch");if(std::memcmp(&c,&reference.inspection[i],sizeof(Car))){std::cerr<<"Mismatch step="<<step<<" vehicle="<<i<<" owner="<<owners[i]<<" actual state/pos="<<c.state<<"/"<<c.pos<<" expected="<<reference.inspection[i].state<<"/"<<reference.inspection[i].pos<<std::endl;throw std::runtime_error("Exact physical equivalence failed");}
 for(auto* p:partitions){bool visible=p->rank==owners[i]||(c.state==1&&jo[model.roads[c.edges[c.k]].from]==p->rank);if(visible){if(std::memcmp(&p->inspection[i],&c,sizeof(Car)))throw std::runtime_error("Ghost state mismatch");}else if(p->inspection[i].state!=-1)throw std::runtime_error("Stale ghost");}
 pending+=c.state==0;stopped+=c.state==1&&c.v<.5f;moving+=c.state==1&&c.v>=.5f;connector+=c.state==2;completed+=c.state==3;if(c.state==1||c.state==2)active[owners[i]]++;}
 validatePhysical(observed,model);auto end=std::chrono::steady_clock::now();double vm=std::chrono::duration<double,std::milli>(end-done).count();verifyMs+=vm;
 series<<step<<','<<t+.25f<<','<<pending<<','<<stopped<<','<<moving<<','<<connector<<','<<completed<<','<<transport.messages-oldMessages<<','<<transport.reservations-oldReservations<<','<<transport.migrations-oldMigrations<<','<<transport.bytes-oldBytes<<','<<transport.controlBytes-oldControl<<','<<2ULL*3*n*4<<','<<active[0]<<','<<active[1]<<','<<active[2]<<','<<std::chrono::duration<double,std::milli>(r-z).count()<<','<<std::chrono::duration<double,std::milli>(done-r).count()<<','<<vm<<'\n';
 if((step+1)%120==0){float time=t+.25f;std::ofstream snap(out+"/snapshot_"+std::to_string(int(time))+".bin",std::ios::binary);snap.write((char*)&n,4);snap.write((char*)&time,4);snap.write((char*)observed.data(),n*sizeof(Car));}
 std::swap(reference.a,reference.b);for(auto* p:partitions)std::swap(p->a,p->b);if((step+1)%120==0)std::cout<<"step "<<step+1<<" exact; transfers "<<transport.migrations<<std::endl;
 }
 std::ofstream final(out+"/final.bin",std::ios::binary);float time=start+steps*.25f;final.write((char*)&n,4);final.write((char*)&time,4);final.write((char*)observed.data(),n*sizeof(Car));
 std::ofstream metrics(out+"/metrics.json");metrics<<"{\"exact_every_step\":true,\"gap_conflict_failures\":0,\"steps\":"<<steps<<",\"requested\":"<<n<<",\"pending\":"<<pending<<",\"road_stopped\":"<<stopped<<",\"road_moving\":"<<moving<<",\"connector\":"<<connector<<",\"completed\":"<<completed<<",\"state_messages\":"<<transport.messages<<",\"reservation_messages\":"<<transport.reservations<<",\"ownership_transfers\":"<<transport.migrations<<",\"wire_bytes\":"<<transport.bytes<<",\"logical_control_bytes\":"<<transport.controlBytes<<",\"actual_owner_map_h2d_bytes\":"<<2ULL*3*n*4*steps<<",\"peak_state_messages\":"<<transport.peak<<",\"state_message_bytes\":"<<sizeof(StateMessage)<<",\"state_pool_capacity_per_partition\":"<<messageCap<<",\"vehicle_slots_per_partition\":"<<n<<",\"reference_wall_ms\":"<<referenceMs<<",\"partition_wall_ms\":"<<partitionMs<<",\"verification_ms\":"<<verifyMs<<",\"index\":\""<<option<<"\",\"transport\":\"host_staged_correctness_prototype\"}\n";
 std::cout<<"PASS "<<steps<<" steps, "<<n<<" vehicles; all physical states bitwise equal\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
