#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <tuple>
#include <cstring>
#include "../artifacts/current/vehicle-allocation-oracle-20261018/advance_tasks.cuh"

// Immutable one-step motion replay. Unselected vehicles remain readable context.
// Region/state filtering changes launch occupancy; these timings are diagnostic,
// not additive pieces of full-run latency.
int main(int argc,char** argv){try{
 if(argc<6)throw std::runtime_error("model snapshot road_regions output mode");
 std::ifstream model(argv[1]);int nr,nm,nj;model>>nr>>nm>>nj;
 std::vector<Road> roads(nr);std::vector<Movement> moves(nm);
 for(auto& r:roads)model>>r.from>>r.to>>r.lanes>>r.length>>r.speed;
 for(auto& m:moves)model>>m.junction>>m.in>>m.out>>m.inLane>>m.outLane>>m.turn>>m.axis>>m.length>>m.rtor;
 std::string format;int entries;model>>format>>entries;if(format!="SPARSE")throw std::runtime_error("Sparse study model required");
 std::vector<std::vector<int>> rows(nm);for(int k=0;k<entries;++k){int a,b;model>>a>>b;rows.at(a).push_back(b);}
 std::vector<int> csr(nm+1);for(int i=0;i<nm;++i)csr[i+1]=csr[i]+rows[i].size();for(auto row:rows)csr.insert(csr.end(),row.begin(),row.end());
 int n;float t;std::ifstream snap(argv[2],std::ios::binary);snap.read((char*)&n,4);snap.read((char*)&t,4);std::vector<Car> cars(n);snap.read((char*)cars.data(),n*sizeof(Car));if(!snap)throw std::runtime_error("Invalid snapshot");
 int mode=atoi(argv[5]);bool exclusive=std::getenv("LPSIM_JUNCTION_EXCLUSIVE")&&std::string(std::getenv("LPSIM_JUNCTION_EXCLUSIVE"))=="1";
 std::ifstream reg(argv[3]);std::vector<int> regions(nr);for(auto& r:regions)reg>>r;
 std::vector<int> junctionRegion(nj,-1);for(auto m:moves)if(junctionRegion[m.junction]<0)junctionRegion[m.junction]=regions[m.in];
 auto* dr=device(roads);auto* dm=device(moves);auto* dc=device(csr);auto* a=device(cars);auto* b=device(cars);
 JunctionSummary summary{};summary.capacity=std::max(nr*8,nm);int cap=summary.capacity;
 summary.nearest=device(std::vector<float>(cap));summary.arrival=device(std::vector<float>(cap));summary.tail=device(std::vector<float>(cap));
 summary.busy=device(std::vector<int>(cap));summary.reserved=device(std::vector<int>(cap));summary.owner=device(std::vector<int>(cap));summary.spawn=device(std::vector<int>(cap));summary.granted=device(std::vector<int>(nm));
 std::vector<int> offsets(nj+1);for(auto m:moves)offsets[m.junction+1]++;for(int j=0;j<nj;++j)offsets[j+1]+=offsets[j];summary.junctionOffsets=device(offsets);
 auto* ds=device(std::vector<JunctionSummary>{summary});EntryRequest* dq;check(cudaMalloc(&dq,n*sizeof(EntryRequest)));
 int *order,*leaders;check(cudaMalloc(&order,n*sizeof(int)));check(cudaMalloc(&leaders,n*sizeof(int)));

 JunctionRequestIndex requestIndex(n,nj,true);
 auto common=[&](){initializeLaneOrder<<<(n+127)/128,128>>>(order,n);thrust::sort(thrust::device_pointer_cast(order),thrust::device_pointer_cast(order)+n,LanePositionLess{a});linkLaneLeaders<<<(n+127)/128,128>>>(a,order,leaders,n);};
 MotionContext motion(nr,nm,n);
 auto pre=[&](){common();motion.build(a,dm,dc,t,mode,exclusive);};
 auto finish=[&](){prepareJunctions<<<(std::max(n,cap)+127)/128,128>>>(b,n,dm,.25,ds);summarizeJunctions<<<(n+127)/128,128>>>(b,n,dr,dm,t,mode,ds);requestJunctionEntry<<<(n+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq);requestIndex.build(dq,dm);arbitrateIndexedJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq,requestIndex.keys,requestIndex.begin,requestIndex.end);spawnJunctionVehicles<<<(nr*8+127)/128,128>>>(b,n,nr,ds);};
 std::vector<int> identity(n);std::iota(identity.begin(),identity.end(),0);int* work=device(identity);unsigned long long* ticks;check(cudaMalloc(&ticks,n*sizeof(*ticks)));
 auto advanceRun=[&](bool direct,int* ids){if(direct)advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);else advanceTasks<false><<<(n+127)/128,128>>>(a,b,n,ids,nullptr,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);check(cudaGetLastError());};
 auto makeSchedule=[&](int policy,const std::vector<Car>& state){std::vector<int> ids=identity;
  if(policy==2)std::stable_sort(ids.begin(),ids.end(),[&](int i,int j){const auto& x=state[i];const auto& y=state[j];return std::make_tuple(regions[x.edges[x.k]],x.edges[x.k],x.lanes[x.k],x.pos,i)<std::make_tuple(regions[y.edges[y.k]],y.edges[y.k],y.lanes[y.k],y.pos,j);});
  if(policy==3){int blocks=(n+127)/128;std::vector<std::vector<int>> buckets(blocks);int at=0;
   for(int pass=0;pass<2;++pass)for(int i=0;i<n;++i)if((state[i].state==1)==(pass==0)){while(int(buckets[at].size())>=std::min(128,n-at*128))at=(at+1)%blocks;buckets[at].push_back(i);at=(at+1)%blocks;}
   ids.clear();for(auto& bucket:buckets)ids.insert(ids.end(),bucket.begin(),bucket.end());}
  if(policy==4)std::stable_sort(ids.begin(),ids.end(),[&](int i,int j){auto key=[&](int z){const auto& c=state[z];int e=c.edges[c.k];return std::make_tuple(c.state,roads[e].length-c.pos<30,c.v<.5f,regions[e],e,c.lanes[c.k],z);};return key(i)<key(j);});
  return ids;};
 auto fit=[&](const std::vector<int>& calibration){check(cudaMemcpy(work,calibration.data(),n*sizeof(int),cudaMemcpyHostToDevice));int nw=(n+31)/32;std::vector<std::vector<double>> samples(nw);std::vector<unsigned long long> observed(n);
  for(int rep=0;rep<5;++rep){advanceTasks<true><<<(n+127)/128,128>>>(a,b,n,work,ticks,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);check(cudaGetLastError());check(cudaMemcpy(observed.data(),ticks,n*sizeof(*ticks),cudaMemcpyDeviceToHost));for(int w=0;w<nw;++w){unsigned long long cost=0;for(int k=w*32;k<std::min(n,(w+1)*32);++k)cost=std::max(cost,observed[k]);samples[w].push_back(cost);}}
  std::vector<double> costs(nw);for(int w=0;w<nw;++w){std::sort(samples[w].begin(),samples[w].end());costs[w]=samples[w][2];}return costs;};
 auto oracle=[&](int policy,const std::vector<int>& calibration,const std::vector<double>& costs){int nw=costs.size(),nb=(nw+3)/4;std::vector<int> rank(nw);std::iota(rank.begin(),rank.end(),0);std::stable_sort(rank.begin(),rank.end(),[&](int x,int y){return costs[x]>costs[y];});std::vector<int> tiles;
  if(policy==5)tiles=rank;else {std::vector<std::vector<int>> buckets(nb);std::vector<double> loads(nb);for(int w:rank){int best=-1;for(int k=0;k<nb;++k)if(int(buckets[k].size())<std::min(4,nw-k*4)&&(best<0||loads[k]<loads[best]))best=k;buckets[best].push_back(w);loads[best]+=costs[w];}for(auto& bucket:buckets)tiles.insert(tiles.end(),bucket.begin(),bucket.end());}
  std::vector<int> ids;for(int w:tiles)for(int k=w*32;k<std::min(n,(w+1)*32);++k)ids.push_back(calibration[k]);return ids;};
 auto permutation=[&](const std::vector<int>& ids){auto sorted=ids;std::sort(sorted.begin(),sorted.end());if(sorted!=identity)throw std::runtime_error("Invalid assignment permutation");};
 std::vector<Car> expectedMotion(n),expectedFinal(n),observed(n);pre();advanceRun(true,nullptr);check(cudaMemcpy(expectedMotion.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));finish();check(cudaMemcpy(expectedFinal.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
 std::vector<float*> fp={summary.nearest,summary.arrival,summary.tail};std::vector<int*> ip={summary.busy,summary.reserved,summary.owner,summary.spawn};std::vector<std::vector<float>> ef(3,std::vector<float>(cap));std::vector<std::vector<int>> ei(4,std::vector<int>(cap));
 for(int k=0;k<3;++k)check(cudaMemcpy(ef[k].data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));for(int k=0;k<4;++k)check(cudaMemcpy(ei[k].data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));
 auto checkCars=[&](const std::vector<Car>& reference){check(cudaMemcpy(observed.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));if(std::memcmp(observed.data(),reference.data(),n*sizeof(Car)))throw std::runtime_error("Exact Car bytes mismatch");};
 auto validate=[&](){checkCars(expectedFinal);std::vector<float> f(cap);std::vector<int> z(cap);for(int k=0;k<3;++k){check(cudaMemcpy(f.data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));if(f!=ef[k])throw std::runtime_error("Float summary mismatch");}for(int k=0;k<4;++k){check(cudaMemcpy(z.data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));if(z!=ei[k])throw std::runtime_error("Reservation summary mismatch");}};
 pre();auto fittingStart=std::chrono::steady_clock::now();auto calibration=makeSchedule(3,cars);auto costs=fit(calibration);double fittingMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-fittingStart).count();
 std::ofstream costfile(std::string(argv[4])+".costs.csv");costfile<<"tile,median_cycles,fit_wall_ms\n";for(int i=0;i<int(costs.size());++i)costfile<<i<<','<<costs[i]<<','<<fittingMs<<'\n';
 std::vector<std::vector<int>> schedules(7);std::vector<int*> fixed(7,nullptr);for(int p=1;p<7;++p){schedules[p]=p<5?makeSchedule(p,cars):oracle(p,calibration,costs);permutation(schedules[p]);fixed[p]=device(schedules[p]);advanceRun(false,fixed[p]);checkCars(expectedMotion);}
 cudaEvent_t events[5];for(auto& e:events)check(cudaEventCreate(&e));std::ofstream out(argv[4]);out<<"policy,charged,repeat,pre_ms,setup_ms,motion_ms,junction_ms,pipeline_ms,wall_ms,exact\n";
 std::mt19937 rng(1801);std::vector<std::pair<int,int>> configs;for(int charged=0;charged<2;++charged)for(int policy=0;policy<7;++policy)configs.push_back({policy,charged});
 for(int rep=-3;rep<10;++rep){std::shuffle(configs.begin(),configs.end(),rng);for(auto config:configs){int p=config.first,charged=config.second;check(cudaDeviceSynchronize());auto started=std::chrono::steady_clock::now();check(cudaEventRecord(events[0]));pre();check(cudaEventRecord(events[1]));int* tasks=fixed[p];std::vector<int> current;
  if(charged&&p){std::vector<Car> state(n);check(cudaMemcpy(state.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));if(p<5)current=makeSchedule(p,state);else {auto cal=makeSchedule(3,state);auto measured=fit(cal);current=oracle(p,cal,measured);}check(cudaMemcpy(work,current.data(),n*sizeof(int),cudaMemcpyHostToDevice));tasks=work;}
  check(cudaEventRecord(events[2]));advanceRun(p==0,tasks);check(cudaEventRecord(events[3]));finish();check(cudaEventRecord(events[4]));check(cudaEventSynchronize(events[4]));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();float elapsed[4],total;for(int k=0;k<4;++k)check(cudaEventElapsedTime(elapsed+k,events[k],events[k+1]));check(cudaEventElapsedTime(&total,events[0],events[4]));validate();
  if(charged&&p){permutation(current);advanceRun(false,tasks);checkCars(expectedMotion);} // Outside timed interval, verify charged motion as well.
  if(rep>=0)out<<p<<','<<charged<<','<<rep<<','<<elapsed[0]<<','<<elapsed[1]<<','<<elapsed[2]<<','<<elapsed[3]<<','<<total<<','<<wall<<",true\n";
 }}
 check(cudaDeviceSynchronize());return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
