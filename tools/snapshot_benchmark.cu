// Compile the production kernel directly, not a rewritten approximation.
#include "../src/simulator/junction_simulator.cu"
#include <numeric>
int main(int argc,char** argv){try{
 if(argc<5)throw std::runtime_error("snapshot_benchmark model snapshot output repeats");
 std::ifstream model(argv[1]);int nr,nm;model>>nr>>nm;std::vector<Road> roads(nr);std::vector<Movement> moves(nm);
 for(auto& r:roads)model>>r.from>>r.to>>r.lanes>>r.length>>r.speed;
 for(auto& m:moves)model>>m.junction>>m.in>>m.out>>m.inLane>>m.outLane>>m.turn>>m.axis>>m.length>>m.rtor;
 std::ifstream input(argv[2]);int n;input>>n;std::vector<Car> cars(n);std::vector<int> ids(n);
 for(int i=0;i<n;++i){auto& c=cars[i];int edge,mid;input>>ids[i]>>c.state>>edge>>c.pos>>c.v>>mid;c.n=2;c.edges[0]=edge;c.edges[1]=moves[mid].out;c.lanes[0]=c.lanes[1]=0;c.moves[0]=mid;c.departure=c.state==0?1000:0;}
 if(!input||!model)throw std::runtime_error("Invalid input");
 // Fixtures contain no connectors; conflict matrix accesses short-circuit on state==2.
 auto* dr=device(roads);auto* dm=device(moves);auto* dc=device(std::vector<int>{0});auto* a=device(cars);auto* b=device(cars);
 cudaEvent_t start,end;check(cudaEventCreate(&start));check(cudaEventCreate(&end));int reps=atoi(argv[4]);std::vector<float> times;
 for(int k=0;k<reps+3;++k){
#ifdef SNAPSHOT_INSTRUMENT
 unsigned long long zero[5]={};check(cudaMemcpyToSymbol(snapshotCounts,zero,sizeof(zero)));
#endif
 check(cudaEventRecord(start));advance<<<(n+127)/128,128>>>(a,b,n,dr,dm,dc,nm,180,.25,2,false);check(cudaGetLastError());check(cudaEventRecord(end));check(cudaEventSynchronize(end));float ms;check(cudaEventElapsedTime(&ms,start,end));if(k>=3)times.push_back(ms);}
 // Input a is immutable: every launch evaluates exactly the same physical step.
 check(cudaMemcpy(cars.data(),b,n*sizeof(Car),cudaMemcpyDeviceToHost));std::vector<int> order(n);std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](int i,int j){return ids[i]<ids[j];});
 std::ofstream out(argv[3]);out<<"id,state,edge,pos,speed\n"<<std::setprecision(9);
 for(int i:order){auto c=cars[i];out<<ids[i]<<','<<c.state<<','<<c.edges[c.k]<<','<<c.pos<<','<<c.v<<'\n';}
 std::ofstream timing(std::string(argv[3])+".timing.json");timing<<"{\"milliseconds\":[";for(size_t k=0;k<times.size();++k)timing<<(k?",":"")<<times[k];timing<<"],\"requested\":"<<n<<",\"car_bytes\":"<<sizeof(Car)<<"}\n";
#ifdef SNAPSHOT_INSTRUMENT
 unsigned long long counts[5];check(cudaMemcpyFromSymbol(counts,snapshotCounts,sizeof(counts)));
 std::ofstream counter(std::string(argv[3])+".counts.json");counter<<"{\"road_updates\":"<<counts[0]<<",\"receiving_candidates\":"<<counts[1]<<",\"leader_candidates\":"<<counts[2]<<",\"connector_candidates\":"<<counts[3]<<",\"stopline_candidates\":"<<counts[4]<<"}\n";
#endif
 check(cudaFree(a));check(cudaFree(b));check(cudaFree(dr));check(cudaFree(dm));check(cudaFree(dc));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
