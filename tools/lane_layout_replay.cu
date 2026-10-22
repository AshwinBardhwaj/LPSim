#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <tuple>
#include <cstring>


// Preserve stable IDs at the junction boundary; only motion storage is permuted.
// Flatten record words so neighboring threads copy adjacent fields. memcpy keeps
// the bitwise copy legal for float and integer fields without aliasing assumptions.
__global__ void copyLaneWords(const Car* src,Car* dst,const int* order,int n,bool scatter){
 int w=blockIdx.x*blockDim.x+threadIdx.x;constexpr int words=sizeof(Car)/sizeof(unsigned int);
 if(w>=n*words)return;int slot=w/words,field=w%words;int mapped=order[slot];unsigned int value;
 int from=scatter?w:mapped*words+field,to=scatter?mapped*words+field:w;
 memcpy(&value,reinterpret_cast<const char*>(src)+from*sizeof(value),sizeof(value));
 memcpy(reinterpret_cast<char*>(dst)+to*sizeof(value),&value,sizeof(value));
}
__global__ void invertLaneOrder(const int* order,int* inverse,int n){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n)inverse[order[k]]=k;}
__global__ void mapLaneLeaders(const int* original,const int* order,const int* inverse,int* mapped,int n){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n){int j=original[order[k]];mapped[k]=j<0?-1:inverse[j];}}
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
 LeaderIndex comparator(n,true,true),radix(n,true,true);LeaderIndex* selected=&comparator;
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
 Car* packed=device(cars);Car* packedNext=device(cars);int* inverse=device(std::vector<int>(n));int* mapped=device(std::vector<int>(n));
 auto pack=[&](){copyLaneWords<<<(n*(sizeof(Car)/4)+255)/256,256>>>(a,packed,selected->order,n,false);invertLaneOrder<<<(n+127)/128,128>>>(selected->order,inverse,n);mapLaneLeaders<<<(n+127)/128,128>>>(selected->leaders,selected->order,inverse,mapped,n);check(cudaGetLastError());};
 auto movePacked=[&](){advance<<<(n+127)/128,128>>>(packed,packedNext,n,dr,dm,dc,nm,t,.25,mode,exclusive,mapped,nullptr,-1,motion.head,motion.links,motion.available);};
 auto unpack=[&](){copyLaneWords<<<(n*(sizeof(Car)/4)+255)/256,256>>>(packedNext,b,selected->order,n,true);check(cudaGetLastError());};
 // All variants use the same radix leader builder and unmodified motion kernel.
 // Supplied layout is an optimistic bound: no cost charged for materializing input.
 common();pack();
 cudaEvent_t ev[7];for(auto& e:ev)check(cudaEventCreate(&e));std::ofstream out(argv[4]);out<<"variant,repeat,index_ms,pack_ms,context_ms,motion_ms,unpack_ms,junction_ms,pipeline_ms,wall_ms,exact\n";
 std::mt19937 rng(1901);const char* names[]={"current","lane_charged","lane_supplied"};
 for(int rep=-3;rep<10;++rep){std::vector<int> variants={0,1,2};std::shuffle(variants.begin(),variants.end(),rng);for(int variant:variants){
 check(cudaDeviceSynchronize());auto begin=std::chrono::steady_clock::now();check(cudaEventRecord(ev[0]));common();check(cudaEventRecord(ev[1]));if(variant==1)pack();check(cudaEventRecord(ev[2]));motion.build(variant?packed:a,dm,dc,t,mode,exclusive);check(cudaEventRecord(ev[3]));if(variant)movePacked();else motionRun(false);check(cudaEventRecord(ev[4]));if(variant)unpack();check(cudaEventRecord(ev[5]));finish();check(cudaEventRecord(ev[6]));check(cudaEventSynchronize(ev[6]));double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();float dt[6],total;for(int j=0;j<6;++j)check(cudaEventElapsedTime(dt+j,ev[j],ev[j+1]));check(cudaEventElapsedTime(&total,ev[0],ev[6]));validate();
 if(rep>=0){out<<names[variant]<<','<<rep;for(float v:dt)out<<','<<v;out<<','<<total<<','<<wall<<",true\n";}
 motion.build(variant?packed:a,dm,dc,t,mode,exclusive);if(variant){movePacked();unpack();}else motionRun(false);checkCars(expectedMotion);
 }}return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
