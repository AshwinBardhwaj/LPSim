#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include "memory_variants.cuh"
#include <random>
#include <tuple>
#include <cstring>


__global__ void countRegionStates(const Car* cars,int n,const int* regions,int* counts){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n){const Car& c=cars[i];if(c.state==1||c.state==2)atomicAdd(counts+regions[c.edges[c.k]]*2+(c.state==2),1);}}
// Immutable one-step replay: change leader-index construction only.
// Preparation, sorting, linking and every unchanged stage are timed separately.
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
 LeaderIndex comparator(n,true,false),radix(n,true,true);LeaderIndex* selected=&comparator;
 int* leaders=selected->leaders;
 JunctionRequestIndex requestIndex(n,nj,true);
 auto common=[&](){selected->build(a);leaders=selected->leaders;};
 MotionContext motion(nr,nm,n);
 auto pre=[&](){common();motion.build(a,dm,dc,t,mode,exclusive);};
 auto finish=[&](){prepareJunctions<<<(std::max(n,cap)+127)/128,128>>>(b,n,dm,.25,ds);summarizeJunctions<<<(n+127)/128,128>>>(b,n,dr,dm,t,mode,ds);requestJunctionEntry<<<(n+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq);requestIndex.build(dq,dm);arbitrateIndexedJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq,requestIndex.keys,requestIndex.begin,requestIndex.end);spawnJunctionVehicles<<<(nr*8+127)/128,128>>>(b,n,nr,ds);};
 auto motionRun=[&](bool unused){advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);check(cudaGetLastError());};
 comparator.build(a);radix.build(a);std::vector<int> x(n),y(n);
 for(int field=0;field<2;++field){check(cudaMemcpy(x.data(),field?comparator.leaders:comparator.order,n*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(y.data(),field?radix.leaders:radix.order,n*sizeof(int),cudaMemcpyDeviceToHost));if(x!=y)throw std::runtime_error("Leader/order mismatch");}
 std::vector<Car> expectedMotion(n),expectedFinal(n),observed(n);pre();motionRun(false);check(cudaMemcpy(expectedMotion.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));finish();check(cudaMemcpy(expectedFinal.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
 std::vector<float*> fp={summary.nearest,summary.arrival,summary.tail};std::vector<int*> ip={summary.busy,summary.reserved,summary.owner,summary.spawn};std::vector<std::vector<float>> ef(3,std::vector<float>(cap));std::vector<std::vector<int>> ei(4,std::vector<int>(cap));
 for(int k=0;k<3;++k)check(cudaMemcpy(ef[k].data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));for(int k=0;k<4;++k)check(cudaMemcpy(ei[k].data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));
 auto checkCars=[&](const std::vector<Car>& ref){check(cudaMemcpy(observed.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));if(std::memcmp(observed.data(),ref.data(),n*sizeof(Car)))throw std::runtime_error("Car mismatch");};
 auto validate=[&](){checkCars(expectedFinal);std::vector<float> f(cap);std::vector<int> z(cap);for(int k=0;k<3;++k){check(cudaMemcpy(f.data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));if(f!=ef[k])throw std::runtime_error("Float summary mismatch");}for(int k=0;k<4;++k){check(cudaMemcpy(z.data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));if(z!=ei[k])throw std::runtime_error("Reservation mismatch");}};

 if(argc<7)throw std::runtime_error("Index policy file required");
 CompactState hot{device(std::vector<float>(14*n)),n};MemoryState memory{hot};
 int regionCount=*std::max_element(regions.begin(),regions.end())+1;int* regionDevice=device(regions);int* counts=device(std::vector<int>(regionCount*2));
 check(cudaDeviceSynchronize());auto featuresBegin=std::chrono::steady_clock::now();check(cudaMemset(counts,0,regionCount*2*sizeof(int)));countRegionStates<<<(n+127)/128,128>>>(a,n,regionDevice,counts);std::vector<int> featureCounts(regionCount*2);check(cudaMemcpy(featureCounts.data(),counts,featureCounts.size()*sizeof(int),cudaMemcpyDeviceToHost));double featureMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-featuresBegin).count();
 std::ofstream features(std::string(argv[4])+".features.csv");features<<"region,road_active,connectors,feature_ms\n";for(int j=0;j<regionCount;++j)features<<j<<','<<featureCounts[j*2]<<','<<featureCounts[j*2+1]<<','<<featureMs<<'\n';
 std::ifstream policy(argv[6]);int budget,policyCount;policy>>budget>>policyCount;int mapBytes=nr*8*sizeof(int),featureBytes=regionCount*2*sizeof(int);int maxSlots=(budget-mapBytes-featureBytes)/8;if(maxSlots<1)throw std::runtime_error("Budget below mapping overhead");
 int* laneMap=device(std::vector<int>(nr*8,-1));int* laneKeys=device(std::vector<int>(maxSlots));int* best=device(std::vector<int>(maxSlots));
 struct Policy{std::string name;std::vector<int> map,keys;double cpu;};std::vector<Policy> policies;
 for(int k=0;k<3+policyCount;++k){auto begin=std::chrono::steady_clock::now();Policy p;p.name=k==0?"compact":k==1?"interface":"empty";p.map.assign(nr*8,-1);std::vector<int> chosen(regionCount,0);if(k>=3){int number;policy>>p.name>>number;for(int j=0;j<number;++j){int region;policy>>region;chosen.at(region)=1;}}
 for(int e=0;e<nr;++e)if(chosen[regions[e]])for(int l=0;l<roads[e].lanes;++l){p.map[e*8+l]=p.keys.size();p.keys.push_back(e*8+l);}if(p.keys.size()>maxSlots)throw std::runtime_error("Policy exceeds memory budget");p.cpu=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();policies.push_back(p);}
 selected=&radix;leaders=selected->leaders;
 cudaEvent_t ev[7];for(auto& e:ev)check(cudaEventCreate(&e));std::ofstream out(argv[4]);out<<"variant,repeat,index_ms,context_ms,extract_ms,nearest_ms,motion_ms,junction_ms,pipeline_ms,wall_ms,setup_ms,feature_ms,bytes,budget,exact\n";
 std::mt19937 rng(1901);for(int rep=-3;rep<10;++rep){std::vector<int> variants(policies.size());std::iota(variants.begin(),variants.end(),0);std::shuffle(variants.begin(),variants.end(),rng);for(int variant:variants){auto& p=policies[variant];bool useBase=variant==0||(variant>=3&&p.keys.empty());bool useMap=!useBase&&variant>=2;memory=MemoryState{hot};check(cudaDeviceSynchronize());auto setupBegin=std::chrono::steady_clock::now();if(useMap){check(cudaMemcpy(laneMap,p.map.data(),mapBytes,cudaMemcpyHostToDevice));if(!p.keys.empty())check(cudaMemcpy(laneKeys,p.keys.data(),p.keys.size()*sizeof(int),cudaMemcpyHostToDevice));memory.laneMap=laneMap;memory.best=best;}check(cudaDeviceSynchronize());double setup=p.cpu+std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setupBegin).count();
 auto begin=std::chrono::steady_clock::now();check(cudaEventRecord(ev[0]));common();check(cudaEventRecord(ev[1]));motion.build(a,dm,dc,t,mode,exclusive);check(cudaEventRecord(ev[2]));extractMemory<<<(n+127)/128,128>>>(a,memory);check(cudaEventRecord(ev[3]));if(!p.keys.empty())buildNearestConnector<<<(p.keys.size()+127)/128,128>>>(a,motion.head,motion.links,laneKeys,best,p.keys.size());check(cudaEventRecord(ev[4]));if(useBase)advanceCompact<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);else advanceMemory<<<(n+127)/128,128>>>(memory,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);check(cudaEventRecord(ev[5]));finish();check(cudaEventRecord(ev[6]));check(cudaEventSynchronize(ev[6]));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();float dt[6],total;for(int j=0;j<6;++j)check(cudaEventElapsedTime(dt+j,ev[j],ev[j+1]));check(cudaEventElapsedTime(&total,ev[0],ev[6]));validate();
 if(rep>=0){out<<p.name<<','<<rep;for(float v:dt)out<<','<<v;out<<','<<total<<','<<wall<<','<<(useMap?setup:0)<<','<<featureMs<<','<<(useMap?mapBytes+featureBytes+p.keys.size()*8:0)<<','<<budget<<",true\n";}
 pre();extractMemory<<<(n+127)/128,128>>>(a,memory);if(!p.keys.empty())buildNearestConnector<<<(p.keys.size()+127)/128,128>>>(a,motion.head,motion.links,laneKeys,best,p.keys.size());if(useBase)advanceCompact<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);else advanceMemory<<<(n+127)/128,128>>>(memory,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);checkCars(expectedMotion);
 }}return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
