// Built exclusively from immutable pre-motion cars; never junction post-motion data.
__global__ void collectMotionContext(const Car* cars,int n,const Movement* moves,
 int* head,int* links,int* blocked,int* busy,int* queried){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const Car& c=cars[i];
 if(c.state==2){const Movement m=moves[c.mid];if(head)links[i]=atomicExch(head+m.in*8+m.inLane,i);if(busy)atomicExch(busy+c.mid,1);}
 if(blocked&&c.state==1){if(c.pos<SPACING+8.f)atomicExch(blocked+c.edges[c.k]*8+c.lanes[c.k],1);if(c.k+1<c.n)atomicExch(queried+c.moves[c.k],1);}
}
__global__ void resolveMotionAvailability(const Movement* moves,const int* csr,int nm,
 float t,int mode,bool exclusive,const int* blocked,const int* busy,const int* queried,int* available){
 int m=blockIdx.x*blockDim.x+threadIdx.x;if(m>=nm)return;available[m]=0;if(!queried[m])return;
 const Movement move=moves[m];if(indication(move,t,mode)!=2||blocked[move.out*8+move.outLane])return;
 for(int k=csr[m];k<csr[m+1];++k){int other=csr[nm+1+k];if(busy[other]&&(exclusive||other!=m))return;}
 available[m]=1;
}
struct MotionContext{
 int *head=nullptr,*links=nullptr,*blocked=nullptr,*busy=nullptr,*queried=nullptr,*available=nullptr;int nr,nm,n,flags;
 MotionContext(int roads,int movements,int vehicles):nr(roads),nm(movements),n(vehicles){
  flags=std::getenv("LPSIM_MOTION_CONTEXT")?atoi(std::getenv("LPSIM_MOTION_CONTEXT")):0;
  if(flags<0||flags>3)throw std::runtime_error("LPSIM_MOTION_CONTEXT must be 0..3");
  auto alloc=[](int*& p,int count){check(cudaMalloc(&p,count*sizeof(int)));};
  if(flags&1){alloc(head,nr*8);alloc(links,n);}if(flags&2){alloc(blocked,nr*8);alloc(busy,nm);alloc(queried,nm);alloc(available,nm);}
 }
 void build(const Car* cars,const Movement* moves,const int* csr,float t,int mode,bool exclusive){
  if(!flags)return;if(head)check(cudaMemset(head,255,nr*8*sizeof(int)));
  if(blocked){check(cudaMemset(blocked,0,nr*8*sizeof(int)));check(cudaMemset(busy,0,nm*sizeof(int)));check(cudaMemset(queried,0,nm*sizeof(int)));}
  collectMotionContext<<<(n+127)/128,128>>>(cars,n,moves,head,links,blocked,busy,queried);
  if(available)resolveMotionAvailability<<<(nm+127)/128,128>>>(moves,csr,nm,t,mode,exclusive,blocked,busy,queried,available);check(cudaGetLastError());
 }
 ~MotionContext(){for(auto p:{head,links,blocked,busy,queried,available})if(p)cudaFree(p);}
};
