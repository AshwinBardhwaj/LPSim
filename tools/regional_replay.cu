#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include "compact_motion.cuh"
#include "regional_kernels.cuh"
#include <random>
#include <tuple>
#include <cstring>


// Immutable one-step replay: change leader-index construction only.
// Preparation, sorting, linking and every unchanged stage are timed separately.
int main(int argc,char** argv){try{
 if(argc<7)throw std::runtime_error("model snapshot road_regions output mode");
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
 selected=&radix;leaders=selected->leaders;CompactState hot{device(std::vector<float>(14*n)),n};
 auto setupStart=std::chrono::steady_clock::now();std::ifstream config(argv[6]);int H,regionsCount,selectedCount;config>>H>>regionsCount>>selectedCount;if(H<1||H>20||selectedCount<0||selectedCount>n)throw std::runtime_error("Invalid region configuration");
 std::vector<int> offsetsRegion{0},regionalIds,workIds,seen(n);
 for(int group=0;group<regionsCount;++group){int count;config>>count;if(count<1||count>128)throw std::runtime_error("Region too large");int edge=-1;for(int k=0;k<count;++k){int id;config>>id;if(id<0||id>=n||seen[id]++||cars[id].state!=1)throw std::runtime_error("Invalid region member");if(k&&cars[id].edges[cars[id].k]!=edge)throw std::runtime_error("Mixed roads");edge=cars[id].edges[cars[id].k];regionalIds.push_back(id);}offsetsRegion.push_back(regionalIds.size());}
 if(!config||regionalIds.size()!=selectedCount)throw std::runtime_error("Incomplete region configuration");for(int i=0;i<n;++i)if(!seen[i])workIds.push_back(i);int workCount=workIds.size();int small=0,medium=0;for(int g=0;g<regionsCount;++g){int size=offsetsRegion[g+1]-offsetsRegion[g];small+=size<=8;medium+=size<=32;}
 auto *dOffsets=device(offsetsRegion),*dRegional=device(regionalIds.empty()?std::vector<int>{0}:regionalIds),*dWork=device(workIds.empty()?std::vector<int>{0}:workIds);MotionCar* cache;check(cudaMalloc(&cache,std::max(1,H*selectedCount)*sizeof(MotionCar)));check(cudaDeviceSynchronize());double coldSetup=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-setupStart).count();
 auto transferStart=std::chrono::steady_clock::now();check(cudaMemcpy(dOffsets,offsetsRegion.data(),offsetsRegion.size()*4,cudaMemcpyHostToDevice));if(selectedCount)check(cudaMemcpy(dRegional,regionalIds.data(),selectedCount*4,cudaMemcpyHostToDevice));if(workCount)check(cudaMemcpy(dWork,workIds.data(),workCount*4,cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());double transfer=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-transferStart).count();
 float initialTime=t;
 auto reset=[&](){check(cudaMemcpy(a,cars.data(),n*sizeof(Car),cudaMemcpyHostToDevice));check(cudaMemcpy(b,cars.data(),n*sizeof(Car),cudaMemcpyHostToDevice));check(cudaDeviceSynchronize());t=initialTime;};
 auto contextSubset=[&](){check(cudaMemset(motion.head,255,nr*8*4));check(cudaMemset(motion.blocked,0,nr*8*4));check(cudaMemset(motion.busy,0,nm*4));check(cudaMemset(motion.queried,0,nm*4));if(workCount)collectMotionContextSubset<<<(workCount+127)/128,128>>>(a,n,dm,motion.head,motion.links,motion.blocked,motion.busy,motion.queried,dWork,workCount);resolveMotionAvailability<<<(nm+127)/128,128>>>(dm,dc,nm,t,mode,exclusive,motion.blocked,motion.busy,motion.queried,motion.available);};
 auto finishSubset=[&](){prepareJunctionsSubset<<<(std::max(workCount,cap)+127)/128,128>>>(b,n,dm,.25,ds,dWork,workCount);if(workCount){summarizeJunctionsSubset<<<(workCount+127)/128,128>>>(b,n,dr,dm,t,mode,ds,dWork,workCount);requestJunctionEntrySubset<<<(workCount+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq,dWork,workCount);}check(cudaMemset(requestIndex.begin,0,nj*4));check(cudaMemset(requestIndex.end,0,nj*4));if(workCount){makeSubsetRequests<<<(workCount+127)/128,128>>>(dq,dm,dWork,workCount,requestIndex.keys);thrust::sort(thrust::device_pointer_cast(requestIndex.keys),thrust::device_pointer_cast(requestIndex.keys)+workCount);findJunctionRequestRanges<<<(workCount+127)/128,128>>>(requestIndex.keys,workCount,requestIndex.begin,requestIndex.end);}arbitrateIndexedJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq,requestIndex.keys,requestIndex.begin,requestIndex.end);spawnJunctionVehicles<<<(nr*8+127)/128,128>>>(b,n,nr,ds);};
 std::vector<std::vector<Car>> reference(H,std::vector<Car>(n));reset();
 for(int h=0;h<H;++h){t=initialTime+h*.25f;pre();extractCompact<<<(n+127)/128,128>>>(a,hot);advanceCompact<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);finish();check(cudaMemcpy(reference[h].data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));std::swap(a,b);}
 std::vector<cudaEvent_t> ev(2*H+3);for(auto& event:ev)check(cudaEventCreate(&event));
 std::ofstream out(argv[4]);out<<"variant,repeat,H,selected,regions,local_ms,index_ms,gpu_ms,wall_ms,transfer_ms,cold_setup_ms,exact\n";
 auto run=[&](int variant,int rep,bool verify){reset();bool local=variant&&selectedCount;auto begin=std::chrono::steady_clock::now();check(cudaEventRecord(ev[0]));if(local){if(variant==1)localRoadSteps<128><<<regionsCount,128>>>(a,dr,dOffsets,dRegional,selectedCount,H,.25f,cache,0,regionsCount);else{if(small)localRoadSteps<8><<<(small+15)/16,128>>>(a,dr,dOffsets,dRegional,selectedCount,H,.25f,cache,0,small);if(medium>small)localRoadSteps<32><<<(medium-small+3)/4,128>>>(a,dr,dOffsets,dRegional,selectedCount,H,.25f,cache,small,medium);if(regionsCount>medium)localRoadSteps<128><<<regionsCount-medium,128>>>(a,dr,dOffsets,dRegional,selectedCount,H,.25f,cache,medium,regionsCount);}}check(cudaEventRecord(ev[1]));
 for(int h=0;h<H;++h){t=initialTime+h*.25f;check(cudaEventRecord(ev[2+2*h]));if(local)buildSubsetLeaders(radix,a,dWork,workCount);else common();check(cudaEventRecord(ev[3+2*h]));
 if(local){contextSubset();if(workCount){extractCompactSubset<<<(workCount+127)/128,128>>>(a,hot,dWork,workCount);advanceCompactSubset<<<(workCount+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available,dWork,workCount);}publishLocal<<<(selectedCount+127)/128,128>>>(b,cache,dRegional,selectedCount,h);finishSubset();}
 else{motion.build(a,dm,dc,t,mode,exclusive);extractCompact<<<(n+127)/128,128>>>(a,hot);advanceCompact<<<(n+127)/128,128>>>(hot,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);finish();}
 check(cudaGetLastError());if(verify){check(cudaMemcpy(observed.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));if(std::memcmp(observed.data(),reference[h].data(),n*sizeof(Car)))throw std::runtime_error("Local region mismatch at step "+std::to_string(h));}std::swap(a,b);
 }
 check(cudaEventRecord(ev.back()));check(cudaEventSynchronize(ev.back()));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();float total,localMs,indexMs=0;check(cudaEventElapsedTime(&total,ev[0],ev.back()));check(cudaEventElapsedTime(&localMs,ev[0],ev[1]));for(int h=0;h<H;++h){float ms;check(cudaEventElapsedTime(&ms,ev[2+2*h],ev[3+2*h]));indexMs+=ms;}
 check(cudaMemcpy(observed.data(),a,n*sizeof(Car),cudaMemcpyDeviceToHost));if(std::memcmp(observed.data(),reference.back().data(),n*sizeof(Car)))throw std::runtime_error("Final region mismatch");
 if(rep>=0)out<<(variant==2?"packed":variant?"regional":"bulk")<<','<<rep<<','<<H<<','<<selectedCount<<','<<regionsCount<<','<<localMs<<','<<indexMs<<','<<total<<','<<wall<<','<<(variant?transfer:0)<<','<<(variant?coldSetup:0)<<",true\n";
 };
 run(1,-9,true);run(2,-9,true);std::mt19937 rng(1901);for(int rep=-3;rep<10;++rep){std::vector<int> variants{0,1,2};std::shuffle(variants.begin(),variants.end(),rng);for(int v:variants)run(v,rep,false);}return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
