#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <tuple>

// Immutable one-step motion replay. Unselected vehicles remain readable context.
// Region/state filtering changes launch occupancy; these timings are diagnostic,
// not additive pieces of full-run latency.
int main(int argc,char** argv){try{
 if(argc<7)throw std::runtime_error("model snapshot road_regions output selector speed_override");
 std::ifstream model(argv[1]);int nr,nm,nj;model>>nr>>nm>>nj;
 std::vector<Road> roads(nr);std::vector<Movement> moves(nm);
 for(auto& r:roads)model>>r.from>>r.to>>r.lanes>>r.length>>r.speed;
 for(auto& m:moves)model>>m.junction>>m.in>>m.out>>m.inLane>>m.outLane>>m.turn>>m.axis>>m.length>>m.rtor;
 std::string format;int entries;model>>format>>entries;if(format!="SPARSE")throw std::runtime_error("Sparse study model required");
 std::vector<std::vector<int>> rows(nm);for(int k=0;k<entries;++k){int a,b;model>>a>>b;rows.at(a).push_back(b);}
 std::vector<int> csr(nm+1);for(int i=0;i<nm;++i)csr[i+1]=csr[i]+rows[i].size();for(auto row:rows)csr.insert(csr.end(),row.begin(),row.end());
 int n;float t;std::ifstream snap(argv[2],std::ios::binary);snap.read((char*)&n,4);snap.read((char*)&t,4);std::vector<Car> cars(n);snap.read((char*)cars.data(),n*sizeof(Car));if(!snap)throw std::runtime_error("Invalid snapshot");
 const int originalCount=n;
 if(std::getenv("LPSIM_PADDING")){int extra=atoi(std::getenv("LPSIM_PADDING"));Car padding=cars[0];padding.state=3;padding.pos=padding.v=0;cars.resize(n+extra,padding);n=cars.size();}
 if(std::getenv("LPSIM_HALF_ACTIVE")){int ordinal=0;for(auto& c:cars)if(c.state==1&&ordinal++%2)c.state=3;}
 std::vector<int> ids(n);std::iota(ids.begin(),ids.end(),0);
 if(const char* ordering=std::getenv("LPSIM_REPLAY_ORDER")){
  std::string order(ordering);if(order=="shuffle"){std::mt19937 rng(17);std::shuffle(ids.begin(),ids.end(),rng);}else if(order=="lane")std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){const auto& x=cars[a];const auto& y=cars[b];return std::make_tuple(x.state,x.edges[x.k],x.lanes[x.k],x.pos,a)<std::make_tuple(y.state,y.edges[y.k],y.lanes[y.k],y.pos,b);});
  auto original=cars;for(int i=0;i<n;++i)cars[i]=original[ids[i]];
 }
 int block=std::getenv("LPSIM_REPLAY_BLOCK")?atoi(std::getenv("LPSIM_REPLAY_BLOCK")):128;if(block!=64&&block!=128&&block!=256)throw std::runtime_error("Unsupported block size");
 std::ifstream reg(argv[3]);std::vector<int> regions(nr);for(auto& r:regions)reg>>r;
 int selector=atoi(argv[5]);float speed=atof(argv[6]);if(speed>=0)for(auto& c:cars)if(c.state==1)c.v=std::min(speed,roads[c.edges[c.k]].speed);
 int mode=std::getenv("LPSIM_STUDY_MODE")?atoi(std::getenv("LPSIM_STUDY_MODE")):0;
 std::vector<int> selected(n,0);int active=0,stopped=0,approach=0,requests=0,connector=0,pending=0,completed=0;double speedSum=0,degree=0;
 for(int i=0;i<n;++i){const Car& c=cars[i];int edge=c.edges[c.k];bool near=c.state==1&&roads[edge].length-c.pos<30;
  bool pick=selector<0||(selector<16?regions[edge]==selector:selector==16?c.state==1&&!near&&c.v>=.5:selector==17?c.state==1&&c.v<.5:selector==18?near:selector==19?c.state==2:selector==20?c.state==0:c.state==3);
  selected[i]=pick?0:1;if(!pick)continue;
  active+=c.state==1;stopped+=c.state==1&&c.v<.5;approach+=near;connector+=c.state==2;pending+=c.state==0;completed+=c.state==3;
  if(c.state==1){speedSum+=c.v;if(c.k+1<c.n){int mid=c.moves[c.k];degree+=csr[mid+1]-csr[mid];int light=indication(moves[mid],t,mode);bool eligible=light==2||(light==0&&moves[mid].turn==-1&&moves[mid].rtor&&c.stopped>=2&&!clearance(t,mode));requests+=roads[edge].length-c.pos<=.11f&&eligible;}}
 }
 auto* a=device(cars);auto* b=device(cars);auto* dr=device(roads);auto* dm=device(moves);auto* dc=device(csr);auto* mask=device(selected);
#ifdef STUDY_INSTRUMENT
 unsigned long long* counts;check(cudaMalloc(&counts,n*3*sizeof(unsigned long long)));check(cudaMemset(counts,0,n*3*sizeof(unsigned long long)));check(cudaMemcpyToSymbol(studyCounts,&counts,sizeof(counts)));
#endif
 int *order,*leaders;check(cudaMalloc(&order,n*sizeof(int)));check(cudaMalloc(&leaders,n*sizeof(int)));
 auto build=[&](){initializeLaneOrder<<<(n+block-1)/block,block>>>(order,n);thrust::sort(thrust::device_pointer_cast(order),thrust::device_pointer_cast(order)+n,LanePositionLess{a});linkLaneLeaders<<<(n+block-1)/block,block>>>(a,order,leaders,n);};
 build();advance<<<(n+block-1)/block,block>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,false,leaders);check(cudaGetLastError());
 std::vector<Car> expected(n);check(cudaMemcpy(expected.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
 cudaEvent_t start,middle,end;check(cudaEventCreate(&start));check(cudaEventCreate(&middle));check(cudaEventCreate(&end));std::vector<float> totalTimes,motionTimes;
 for(int repeat=0;repeat<13;++repeat){check(cudaEventRecord(start));build();check(cudaEventRecord(middle));advance<<<(n+block-1)/block,block>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,false,leaders,selector<0?nullptr:mask,0);check(cudaGetLastError());check(cudaEventRecord(end));check(cudaEventSynchronize(end));float total,motion;check(cudaEventElapsedTime(&total,start,end));check(cudaEventElapsedTime(&motion,middle,end));if(repeat>=3){totalTimes.push_back(total);motionTimes.push_back(motion);}}
 std::vector<Car> output(n);check(cudaMemcpy(output.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
#ifdef STUDY_INSTRUMENT
 std::vector<unsigned long long> observed(n*3);std::vector<int> hostLeaders(n);check(cudaMemcpy(hostLeaders.data(),leaders,n*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(observed.data(),counts,n*3*sizeof(unsigned long long),cudaMemcpyDeviceToHost));std::ofstream counters(std::string(argv[4])+".counts.csv");counters<<"id,region,state,speed,distance_to_stop,leader,receiving_candidates,connector_candidates,thread_cycles\n";
 for(int i=0;i<n;++i){const Car& c=cars[i];counters<<i<<','<<regions[c.edges[c.k]]<<','<<c.state<<','<<c.v<<','<<roads[c.edges[c.k]].length-c.pos<<','<<hostLeaders[i]<<','<<observed[i*3]<<','<<observed[i*3+1]<<','<<observed[i*3+2]<<'\n';}check(cudaFree(counts));
#endif
 for(int i=0;i<n;++i)if(selected[i]==0){auto x=output[i],y=expected[i];if(x.state!=y.state||x.pos!=y.pos||x.v!=y.v)throw std::runtime_error("Selected physical output mismatch");}
 unsigned long long digest=1469598103934665603ULL;
 std::vector<int> canonical(n);for(int i=0;i<n;++i)canonical[ids[i]]=i;
 for(int id=0;id<originalCount;++id){const auto& c=output[canonical[id]];float values[]={float(c.state),float(c.k),c.pos,c.v,float(c.event),float(c.denial),c.stopped,c.entryWait};for(unsigned char byte:reinterpret_cast<const unsigned char(&)[sizeof(values)]>(values)){digest^=byte;digest*=1099511628211ULL;}}
 std::ofstream f(argv[4]);f<<std::setprecision(9)<<"{\"time\":"<<t<<",\"selector\":"<<selector<<",\"active\":"<<active<<",\"stopped\":"<<stopped<<",\"approach\":"<<approach<<",\"requests\":"<<requests<<",\"connector\":"<<connector<<",\"pending\":"<<pending<<",\"completed\":"<<completed<<",\"speed_sum\":"<<speedSum<<",\"conflict_degree_sum\":"<<degree<<",\"physical_match\":true,\"total_ms\":[";
 for(int i=0;i<10;++i)f<<(i?",":"")<<totalTimes[i];f<<"],\"motion_ms\":[";for(int i=0;i<10;++i)f<<(i?",":"")<<motionTimes[i];f<<"],\"physical_digest\":"<<digest<<"}\n";
 for(auto p:{a,b})check(cudaFree(p));check(cudaFree(dr));check(cudaFree(dm));for(auto p:{dc,mask,order,leaders})check(cudaFree(p));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
