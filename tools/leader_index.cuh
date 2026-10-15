// Experimental index: rebuild from the immutable input at every measured step.
// Sorting IDs leaves Car storage, launch order, and stable priorities unchanged.
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
struct LanePositionLess {
 const Car* cars;
 __device__ bool operator()(int a,int b) const {
  const Car& x=cars[a];const Car& y=cars[b];
  bool xa=x.state==1,ya=y.state==1;
  if(xa!=ya)return xa;
  if(!xa)return a<b;
  if(x.edges[x.k]!=y.edges[y.k])return x.edges[x.k]<y.edges[y.k];
  if(x.lanes[x.k]!=y.lanes[y.k])return x.lanes[x.k]<y.lanes[y.k];
  if(x.pos!=y.pos)return x.pos<y.pos;
  return a<b;
 }
};
__global__ void initializeLaneOrder(int* order,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)order[i]=i;
}
__global__ void linkLaneLeaders(const Car* cars,const int* order,int* leaders,int n){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;
 int i=order[k];leaders[i]=-1;const Car& c=cars[i];
 if(c.state!=1||k+1>=n)return;
 int j=order[k+1];const Car& d=cars[j];
 if(d.state==1&&d.edges[d.k]==c.edges[c.k]&&d.lanes[d.k]==c.lanes[c.k]&&d.pos>c.pos)leaders[i]=j;
}
