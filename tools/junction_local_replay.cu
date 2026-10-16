#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <tuple>

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
 initializeLaneOrder<<<(n+127)/128,128>>>(order,n);thrust::sort(thrust::device_pointer_cast(order),thrust::device_pointer_cast(order)+n,LanePositionLess{a});linkLaneLeaders<<<(n+127)/128,128>>>(a,order,leaders,n);
 MotionContext motion(nr,nm,n);motion.build(a,dm,dc,t,mode,exclusive);
 advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,t,.25,mode,exclusive,leaders,nullptr,-1,motion.head,motion.links,motion.available);
 prepareJunctions<<<(std::max(n,cap)+127)/128,128>>>(b,n,dm,.25,ds);summarizeJunctions<<<(n+127)/128,128>>>(b,n,dr,dm,t,mode,ds);requestJunctionEntry<<<(n+127)/128,128>>>(b,n,dr,dm,dc,nm,t,mode,exclusive,ds,dq);
 std::vector<Car> before(n),expected(n);std::vector<EntryRequest> requests(n);check(cudaMemcpy(before.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));check(cudaMemcpy(requests.data(),dq,n*sizeof(EntryRequest),cudaMemcpyDeviceToHost));
 std::vector<float*> floatPointers={summary.nearest,summary.arrival,summary.tail};std::vector<int*> intPointers={summary.busy,summary.reserved,summary.owner,summary.spawn};
 std::vector<std::vector<float>> floats(3,std::vector<float>(cap));std::vector<std::vector<int>> ints(4,std::vector<int>(cap));
 for(int k=0;k<3;++k)check(cudaMemcpy(floats[k].data(),floatPointers[k],cap*sizeof(float),cudaMemcpyDeviceToHost));
 for(int k=0;k<4;++k)check(cudaMemcpy(ints[k].data(),intPointers[k],cap*sizeof(int),cudaMemcpyDeviceToHost));
 auto restore=[&](){check(cudaMemcpy(b,before.data(),n*sizeof(Car),cudaMemcpyHostToDevice));for(int k=0;k<3;++k)check(cudaMemcpy(floatPointers[k],floats[k].data(),cap*sizeof(float),cudaMemcpyHostToDevice));for(int k=0;k<4;++k)check(cudaMemcpy(intPointers[k],ints[k].data(),cap*sizeof(int),cudaMemcpyHostToDevice));};
 arbitrateJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq);check(cudaMemcpy(expected.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
 auto expectedFloats=floats;auto expectedInts=ints;
 for(int k=0;k<3;++k)check(cudaMemcpy(expectedFloats[k].data(),floatPointers[k],cap*sizeof(float),cudaMemcpyDeviceToHost));
 for(int k=0;k<4;++k)check(cudaMemcpy(expectedInts[k].data(),intPointers[k],cap*sizeof(int),cudaMemcpyDeviceToHost));
 auto same=[](const Car& x,const Car& y){return x.state==y.state&&x.k==y.k&&x.mid==y.mid&&x.pos==y.pos&&x.v==y.v&&x.event==y.event&&x.denial==y.denial&&x.stopped==y.stopped&&x.entryWait==y.entryWait;};
 JunctionRequestIndex index(n,nj,true);index.build(dq,dm);std::vector<int> begin(nj),end(nj);std::vector<unsigned long long> keys(n);
 check(cudaMemcpy(begin.data(),index.begin,nj*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(end.data(),index.end,nj*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(keys.data(),index.keys,n*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
 // Verify exact membership and priority, including blocked and right-on-red requests.
 std::vector<int> active;for(int j=0;j<nj;++j){std::vector<int> reference;for(int pass=0;pass<2;++pass)for(int i=0;i<n;++i)if(requests[i].pass==pass&&moves[requests[i].mid].junction==j)reference.push_back(i);
  if(int(reference.size())!=end[j]-begin[j])throw std::runtime_error("Request count mismatch");for(int k=0;k<int(reference.size());++k)if(static_cast<unsigned int>(keys[begin[j]+k])!=reference[k])throw std::runtime_error("Priority mismatch");if(!reference.empty())active.push_back(j);
 }
 std::ofstream features(std::string(argv[4])+".junctions.csv");features<<"junction,region,requests,green,right_on_red,preblocked,conflict_edges,admitted,denied\n";
 for(int j:active){int green=0,red=0,blocked=0,degree=0,admitted=0,denied=0;for(int k=begin[j];k<end[j];++k){int i=static_cast<unsigned int>(keys[k]);auto q=requests[i];green+=q.pass==0;red+=q.pass==1;blocked+=q.blockedReason!=0;degree+=csr[q.mid+1]-csr[q.mid];admitted+=expected[i].state==2;denied+=expected[i].denial!=0;}features<<j<<','<<junctionRegion[j]<<','<<end[j]-begin[j]<<','<<green<<','<<red<<','<<blocked<<','<<degree<<','<<admitted<<','<<denied<<'\n';}
 cudaEvent_t start,middle,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&middle));check(cudaEventCreate(&stop));
 std::ofstream out(argv[4]);out<<"selector,junctions,requests,repeat,index_ms,arbitration_ms,total_ms,physical_match\n";
 std::vector<Car> observed(n);
 // -2 global reference; -1 indexed full grid; 16 compact all active; 0..15 local regions.
 for(int selector=-2;selector<=16;++selector){std::vector<int> selected;for(int j:active)if(selector<0||selector==16||junctionRegion[j]==selector)selected.push_back(j);
  int* ids=selected.empty()?nullptr:device(selected);int requestCount=0;for(int j:selected)requestCount+=end[j]-begin[j];
  for(int repeat=0;repeat<13;++repeat){restore();check(cudaEventRecord(start));if(selector==-1)index.build(dq,dm);check(cudaEventRecord(middle));
   if(selector==-2)arbitrateJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq);
   else if(selector==-1)arbitrateIndexedJunctions<<<(nj+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq,index.keys,index.begin,index.end);
   else if(!selected.empty())arbitrateIndexedJunctions<<<(selected.size()+127)/128,128>>>(b,n,dr,dm,dc,nm,nj,exclusive,ds,dq,index.keys,index.begin,index.end,ids,selected.size());
   check(cudaGetLastError());check(cudaEventRecord(stop));check(cudaEventSynchronize(stop));float im,am,total;check(cudaEventElapsedTime(&im,start,middle));check(cudaEventElapsedTime(&am,middle,stop));check(cudaEventElapsedTime(&total,start,stop));
   if(repeat>=3)out<<selector<<','<<selected.size()<<','<<requestCount<<','<<repeat-3<<','<<im<<','<<am<<','<<total<<",true\n";
  }
  check(cudaMemcpy(observed.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));
  for(int i=0;i<n;++i){bool selectedCar=selector<0||selector==16||(requests[i].pass>=0&&junctionRegion[moves[requests[i].mid].junction]==selector);if(!same(observed[i],selectedCar?expected[i]:before[i]))throw std::runtime_error("Local physical mismatch");}
  auto owns=[&](int j){return selector<0||selector==16||(j>=0&&junctionRegion[j]==selector);};
  std::vector<float> fv(cap);std::vector<int> iv(cap);
  for(int k=0;k<3;++k){check(cudaMemcpy(fv.data(),floatPointers[k],cap*sizeof(float),cudaMemcpyDeviceToHost));for(int q=0;q<cap;++q){int owner=k==2&&q<nm?moves[q].junction:-1;float target=owns(owner)?expectedFloats[k][q]:floats[k][q];if(fv[q]!=target)throw std::runtime_error("Local float summary mismatch");}}
  for(int k=0;k<4;++k){check(cudaMemcpy(iv.data(),intPointers[k],cap*sizeof(int),cudaMemcpyDeviceToHost));for(int q=0;q<cap;++q){int owner=(k==1||k==2)&&q<nr*8?roads[q/8].from:-1;int target=owns(owner)?expectedInts[k][q]:ints[k][q];if(iv[q]!=target)throw std::runtime_error("Local reservation summary mismatch");}}
  if(ids)check(cudaFree(ids));
 }
 check(cudaDeviceSynchronize());return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
