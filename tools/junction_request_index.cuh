// Included after simulator types and Thrust headers. No car storage is reordered.
__global__ void makeJunctionRequestKeys(const EntryRequest* requests,const Movement* moves,unsigned long long* keys,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;const EntryRequest r=requests[i];
 keys[i]=r.pass<0?~0ULL:(static_cast<unsigned long long>(moves[r.mid].junction)<<33)|(static_cast<unsigned long long>(r.pass)<<32)|static_cast<unsigned int>(i);
}
__global__ void findJunctionRequestRanges(const unsigned long long* keys,int n,int* begin,int* end){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n||keys[k]==~0ULL)return;int junction=keys[k]>>33;
 if(k==0||(keys[k-1]>>33)!=static_cast<unsigned long long>(junction))begin[junction]=k;
 if(k+1==n||(keys[k+1]>>33)!=static_cast<unsigned long long>(junction))end[junction]=k+1;
}
struct JunctionRequestIndex{
 bool enabled;int n,nj;unsigned long long* keys=nullptr;int *begin=nullptr,*end=nullptr;
 JunctionRequestIndex(int vehicles,int junctions,bool force=false):n(vehicles),nj(junctions){
  enabled=force||(std::getenv("LPSIM_JUNCTION_REQUEST_INDEX")&&std::string(std::getenv("LPSIM_JUNCTION_REQUEST_INDEX"))=="1");
  if(enabled){check(cudaMalloc(&keys,n*sizeof(*keys)));check(cudaMalloc(&begin,nj*sizeof(int)));check(cudaMalloc(&end,nj*sizeof(int)));}
 }
 void build(const EntryRequest* requests,const Movement* moves){
  check(cudaMemset(begin,0,nj*sizeof(int)));check(cudaMemset(end,0,nj*sizeof(int)));
  makeJunctionRequestKeys<<<(n+127)/128,128>>>(requests,moves,keys,n);
  thrust::sort(thrust::device_pointer_cast(keys),thrust::device_pointer_cast(keys)+n);
  findJunctionRequestRanges<<<(n+127)/128,128>>>(keys,n,begin,end);check(cudaGetLastError());
 }
 ~JunctionRequestIndex(){if(keys)cudaFree(keys);if(begin)cudaFree(begin);if(end)cudaFree(end);}
};
