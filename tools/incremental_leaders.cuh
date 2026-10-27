#pragma once
// Canonical IDs never change. Membership is retained; only touched lanes are repaired.
__device__ int roadLane(const Car& c){return c.state==1?c.edges[c.k]*8+c.lanes[c.k]:-1;}
__global__ void seedLaneLists(const Car* cars,const int* order,int n,int* heads,int* next,int* membership){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=n)return;int i=order[k],lane=roadLane(cars[i]);membership[i]=lane;next[i]=-1;if(lane<0)return;
 if(k==0||roadLane(cars[order[k-1]])!=lane)heads[lane]=i;
 if(k+1<n&&roadLane(cars[order[k+1]])==lane)next[i]=order[k+1];
}
__global__ void detectLaneChanges(const Car* cars,int n,const int* membership,int* dirty,int* arrivals,int* arrivalNext,int* leaders,unsigned int* stats){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int lane=roadLane(cars[i]),prior=membership[i];if(lane==prior)return;
 if(stats)atomicAdd(stats,1u);if(prior>=0)atomicExch(dirty+prior,1);if(lane>=0){atomicExch(dirty+lane,1);arrivalNext[i]=atomicExch(arrivals+lane,i);}leaders[i]=-1;
}
__device__ bool positionBefore(const Car* cars,int a,int b){return cars[a].pos<cars[b].pos||(cars[a].pos==cars[b].pos&&a<b);}
__global__ void pruneLaneLists(const Car* cars,int lanes,const int* dirty,int* heads,int* next,const int* arrivals,const int* arrivalNext,int* leaders,unsigned int* stats){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes||!dirty[lane])return;if(stats)atomicAdd(stats+1,1u);
 int head=heads[lane],prev=-1,cur=head;
 while(cur>=0){int following=next[cur];if(roadLane(cars[cur])!=lane){if(prev<0)head=following;else next[prev]=following;}else prev=cur;cur=following;}
 heads[lane]=head;
}
__global__ void repairLaneLists(const Car* cars,int lanes,const int* dirty,int* heads,int* next,const int* arrivals,const int* arrivalNext,int* leaders,unsigned int* stats){
 int lane=blockIdx.x*blockDim.x+threadIdx.x;if(lane>=lanes||!dirty[lane])return;
 int head=heads[lane],prev=-1,cur=head;
 // Arrival list order may be nondeterministic; sorted insertion restores position/ID order.
 for(int entrant=arrivals[lane];entrant>=0;entrant=arrivalNext[entrant]){prev=-1;cur=head;while(cur>=0&&positionBefore(cars,cur,entrant)){prev=cur;cur=next[cur];}next[entrant]=cur;if(prev<0)head=entrant;else next[prev]=entrant;}
 heads[lane]=head;
 for(cur=head;cur>=0;cur=next[cur]){int lead=next[cur];leaders[cur]=lead>=0&&cars[lead].pos>cars[cur].pos?lead:-1;}
}
__global__ void retainMembership(const Car* cars,int n,int* membership){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)membership[i]=roadLane(cars[i]);}
struct IncrementalLeaders {
 LeaderIndex bulk;int n,lanes;bool ready=false;int *heads,*next,*membership,*dirty,*arrivals,*arrivalNext;unsigned int* stats=nullptr;
 IncrementalLeaders(int count,int roads,bool instrument):bulk(count,true,true),n(count),lanes(roads*8){heads=device(std::vector<int>(lanes,-1));next=device(std::vector<int>(n,-1));membership=device(std::vector<int>(n,-1));dirty=device(std::vector<int>(lanes));arrivals=device(std::vector<int>(lanes,-1));arrivalNext=device(std::vector<int>(n,-1));if(instrument)stats=device(std::vector<unsigned int>(2));}
 void build(const Car* cars,bool force=false){
  if(stats)check(cudaMemset(stats,0,8));
  if(!ready||force){bulk.build(cars);check(cudaMemset(heads,255,lanes*4));seedLaneLists<<<(n+127)/128,128>>>(cars,bulk.order,n,heads,next,membership);ready=true;}
  else{check(cudaMemset(dirty,0,lanes*4));check(cudaMemset(arrivals,255,lanes*4));detectLaneChanges<<<(n+127)/128,128>>>(cars,n,membership,dirty,arrivals,arrivalNext,bulk.leaders,stats);pruneLaneLists<<<(lanes+127)/128,128>>>(cars,lanes,dirty,heads,next,arrivals,arrivalNext,bulk.leaders,stats);repairLaneLists<<<(lanes+127)/128,128>>>(cars,lanes,dirty,heads,next,arrivals,arrivalNext,bulk.leaders,stats);retainMembership<<<(n+127)/128,128>>>(cars,n,membership);}
  check(cudaGetLastError());
 }
};
