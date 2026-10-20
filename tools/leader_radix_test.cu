#define LPSIM_SCALABLE_LIBRARY
#include "../src/simulator/scalable_junction_simulator.cu"
#include <random>
#include <numeric>
int main(){try{std::mt19937 rng(1901);int checks=0;
 for(int n:{1,7,128,129,2048,16384}){std::vector<Car> cars(n);LeaderIndex comp(n,true,false),radix(n,true,true);Car* d=device(cars);std::cout<<"n="<<n<<" radix_extra_bytes="<<(20ull*n+radix.temporaryBytes)<<" scratch_bytes="<<radix.temporaryBytes<<"\n";
 for(int step=0;step<16;++step){for(int i=0;i<n;++i){Car& c=cars[i];c.state=step==0?0:step==1?1:rng()%4;c.k=rng()%3;c.edges[c.k]=rng()%5013;c.lanes[c.k]=rng()%8;c.pos=float(int(rng()%50)-10);if(i%5==0){c.state=1;c.edges[c.k]=7;c.lanes[c.k]=2;c.pos=i%2?0.f:-0.f;}}
 if(step==0)for(auto& c:cars)c.state=0;
 check(cudaMemcpy(d,cars.data(),n*sizeof(Car),cudaMemcpyHostToDevice));comp.build(d);radix.build(d);std::vector<int> expected(n),x(n),y(n),leaders(n,-1);std::iota(expected.begin(),expected.end(),0);
 std::sort(expected.begin(),expected.end(),[&](int a,int b){auto& x=cars[a];auto& y=cars[b];if((x.state==1)!=(y.state==1))return x.state==1;if(x.state!=1)return a<b;if(x.edges[x.k]!=y.edges[y.k])return x.edges[x.k]<y.edges[y.k];if(x.lanes[x.k]!=y.lanes[y.k])return x.lanes[x.k]<y.lanes[y.k];if(x.pos!=y.pos)return x.pos<y.pos;return a<b;});
 for(int k=0;k+1<n;++k){int i=expected[k],j=expected[k+1];auto& x=cars[i];auto& y=cars[j];if(x.state==1&&y.state==1&&x.edges[x.k]==y.edges[y.k]&&x.lanes[x.k]==y.lanes[y.k]&&y.pos>x.pos)leaders[i]=j;}
 for(int f=0;f<2;++f){check(cudaMemcpy(x.data(),f?comp.leaders:comp.order,n*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(y.data(),f?radix.leaders:radix.order,n*sizeof(int),cudaMemcpyDeviceToHost));if(x!=y||x!=(f?leaders:expected))throw std::runtime_error("CPU/comparator/radix mismatch");++checks;}
 }check(cudaFree(d));}std::cout<<checks<<" exact CPU/comparator/radix checks passed\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
