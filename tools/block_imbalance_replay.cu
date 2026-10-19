#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <tuple>
#include <cstring>
#include "../artifacts/current/berkeley-block-imbalance-20261019/advance_blocks.cuh"

// Immutable one-step replay of original order, with separate block-timing instrumentation.
// Only assignment changes; all records and physical calculations remain present.
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
 cudaFuncAttributes directAttr{},instAttr{};check(cudaFuncGetAttributes(&directAttr,advance));check(cudaFuncGetAttributes(&instAttr,advanceBlocks));int directResident=0,instResident=0;check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&directResident,advance,128,0));check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&instResident,advanceBlocks,128,0));
 std::ofstream resources(std::string(argv[4])+".resources.csv");resources<<"variant,registers,shared_bytes,local_bytes,max_resident_blocks_per_sm\n";resources<<"original,"<<directAttr.numRegs<<','<<directAttr.sharedSizeBytes<<','<<directAttr.localSizeBytes<<','<<directResident<<'\n';resources<<"instrumented,"<<instAttr.numRegs<<','<<instAttr.sharedSizeBytes<<','<<instAttr.localSizeBytes<<','<<instResident<<'\n';
 if(n%128)throw std::runtime_error("Instrumentation requires full blocks");int nb=n/128;
 unsigned long long *starts,*ends;unsigned int* sms;check(cudaMalloc(&starts,nb*sizeof(*starts)));check(cudaMalloc(&ends,nb*sizeof(*ends)));check(cudaMalloc(&sms,nb*sizeof(*sms)));
 auto motionRun=[&](bool instrument){if(instrument)advanceBlocks<<<nb,128>>>(a,b,n,starts,ends,sms,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);else advance<<<nb,128>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);check(cudaGetLastError());};
 std::vector<Car> expectedMotion(n),expectedFinal(n),observed(n);pre();motionRun(false);check(cudaMemcpy(expectedMotion.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));finish();check(cudaMemcpy(expectedFinal.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
 std::vector<float*> fp={summary.nearest,summary.arrival,summary.tail};std::vector<int*> ip={summary.busy,summary.reserved,summary.owner,summary.spawn};std::vector<std::vector<float>> ef(3,std::vector<float>(cap));std::vector<std::vector<int>> ei(4,std::vector<int>(cap));
 for(int k=0;k<3;++k)check(cudaMemcpy(ef[k].data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));for(int k=0;k<4;++k)check(cudaMemcpy(ei[k].data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));
 auto checkCars=[&](const std::vector<Car>& ref){check(cudaMemcpy(observed.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));if(std::memcmp(observed.data(),ref.data(),n*sizeof(Car)))throw std::runtime_error("Car mismatch");};
 auto validate=[&](){checkCars(expectedFinal);std::vector<float> f(cap);std::vector<int> z(cap);for(int k=0;k<3;++k){check(cudaMemcpy(f.data(),fp[k],cap*sizeof(float),cudaMemcpyDeviceToHost));if(f!=ef[k])throw std::runtime_error("Float summary mismatch");}for(int k=0;k<4;++k){check(cudaMemcpy(z.data(),ip[k],cap*sizeof(int),cudaMemcpyDeviceToHost));if(z!=ei[k])throw std::runtime_error("Reservation mismatch");}};
 cudaEvent_t ev[5];for(auto& e:ev)check(cudaEventCreate(&e));std::ofstream out(argv[4]);out<<"instrument,repeat,pre_ms,leader_ms,context_ms,motion_ms,junction_ms,pipeline_ms,wall_ms,exact\n";
 std::ofstream blocks(std::string(argv[4])+".blocks.csv");blocks<<"repeat,block,sm,start_ns,end_ns\n";std::vector<unsigned long long> hs(nb),he(nb);std::vector<unsigned int> hm(nb);
 // Keep instrumented observations separate from clean timing repetitions.
 for(int inst=0;inst<2;++inst)for(int rep=-3;rep<10;++rep){check(cudaDeviceSynchronize());auto begin=std::chrono::steady_clock::now();check(cudaEventRecord(ev[0]));common();check(cudaEventRecord(ev[1]));motion.build(a,dm,dc,t,mode,exclusive);check(cudaEventRecord(ev[2]));motionRun(inst);check(cudaEventRecord(ev[3]));finish();check(cudaEventRecord(ev[4]));check(cudaEventSynchronize(ev[4]));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();float dt[4],total;for(int j=0;j<4;++j)check(cudaEventElapsedTime(dt+j,ev[j],ev[j+1]));check(cudaEventElapsedTime(&total,ev[0],ev[4]));validate();
  if(rep>=0){out<<inst<<','<<rep<<','<<(dt[0]+dt[1])<<','<<dt[0]<<','<<dt[1]<<','<<dt[2]<<','<<dt[3]<<','<<total<<','<<wall<<",true\n";if(inst){check(cudaMemcpy(hs.data(),starts,nb*sizeof(*starts),cudaMemcpyDeviceToHost));check(cudaMemcpy(he.data(),ends,nb*sizeof(*ends),cudaMemcpyDeviceToHost));check(cudaMemcpy(hm.data(),sms,nb*sizeof(*sms),cudaMemcpyDeviceToHost));for(int k=0;k<nb;++k){if(he[k]<hs[k])throw std::runtime_error("Timestamp reversal");blocks<<rep<<','<<k<<','<<hm[k]<<','<<hs[k]<<','<<he[k]<<'\n';}}}
  // Pure motion equivalence outside the timed interval.
  pre();motionRun(inst);checkCars(expectedMotion);
 }
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
