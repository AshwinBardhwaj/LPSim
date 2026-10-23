#pragma once
// Field-wise shared cache: consecutive threads address consecutive bank words.
struct SharedState:MemoryState {
 float* soaCache=nullptr;
 __device__ MotionCar load(int id) const {
  if(soaCache){int s=cacheInverse[id]-cacheBase;if(s>=0&&s<cacheCount)return CompactState{soaCache,128}.load(s);}
  return MemoryState::load(id);
 }
};
__global__ void advanceSharedSoA(SharedState old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive,const int* leaders=nullptr,const int* work=nullptr,int region=-1,const int* connectorHead=nullptr,const int* connectorNext=nullptr,const int* movementOpen=nullptr){
 __shared__ float local[14*128];int slot=blockIdx.x*blockDim.x+threadIdx.x;
 int i=slot<n?old.order[slot]:0;if(slot<n)CompactState{local,128}.store(threadIdx.x,old.load(i));__syncthreads();if(slot>=n)return;
 old.soaCache=local;old.cacheBase=blockIdx.x*128;old.cacheCount=min(128,n-old.cacheBase);if(work&&work[i]!=region){writeCompact(next[i],old.load(i));return;}MotionCar c=old.load(i);c.event=0;c.denial=0;
 if(c.state==1){const int edge=c.edge,lane=c.lane;const Road r=roads[edge];
  bool open=c.k==c.n-1;
  if(c.k+1<c.n){int mid=c.movement;const Movement m=moves[mid];open=indication(m,t,mode)==2;
   if(movementOpen)open=movementOpen[mid];
   else if(open)for(int j=0;j<n;++j){const MotionCar d=old.load(j);
    if((d.state==2&&conflictsWith(conflict,nm,mid,d.mid)&&(exclusive||mid!=d.mid))||
       (d.state==1&&d.edge==m.out&&d.lane==m.outLane&&d.pos<SPACING+8.f)){open=false;break;}}
  }
  float gap=open?1.e6f:r.length-c.pos+2,leaderV=0,bound=r.length;
  if(leaders){if(leaders[i]>=0){const MotionCar d=old.load(leaders[i]);float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}else for(int j=0;j<n;++j){if(j==i)continue;const MotionCar d=old.load(j);if(d.state==1&&d.edge==edge&&d.lane==lane&&d.pos>c.pos){float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}
  int chosenConnector=-1;int entry=old.laneMap?old.laneMap[edge*8+lane]:-1;int best=entry>=0?old.best[entry]:-2;
  for(int j=best!=-2?best:(connectorHead?connectorHead[edge*8+lane]:0);j>=0&&j<n;j=best!=-2?-1:(connectorHead?connectorNext[j]:j+1)){const MotionCar d=old.load(j);if(d.state==2&&moves[d.mid].in==edge&&moves[d.mid].inLane==lane){
   float distance=r.length-c.pos+d.pos;float g=distance-4.5f;
   // Preserve the original ascending-ID tie choice despite atomic list order.
   if(g<gap||(g==gap&&chosenConnector>=0&&j<chosenConnector)){gap=g;leaderV=d.v;chosenConnector=j;}
   bound=fminf(bound,r.length+d.pos-SPACING);
  }}
  float desired=1.38075f+fmaxf(0.f,c.v*.5433028f+c.v*(c.v-leaderV)/(2*sqrtf(.5570409f*2.902058f)));
  float acc=.5570409f*(1-powf(c.v/r.speed,4)-powf(desired/fmaxf(.05f,gap),2));
  float speed=fmaxf(0.f,c.v+acc*dt);float proposed=c.pos+.5f*(c.v+speed)*dt;
  if(proposed>bound){proposed=fmaxf(c.pos,bound);speed=0;}
  c.pos=proposed;c.v=speed;
  if(r.length-c.pos<.1f && c.v<.3f){c.pos=r.length;c.v=0;}
  c.stopped=(c.v<.1f&&r.length-c.pos<.11f)?c.stopped+dt:0;
  if(c.k==c.n-1&&r.length-c.pos<.11f)c.state=3;
 }
 writeCompact(next[i],c);
}
