// Generated motion variants; the equations remain the compact baseline equations.
#pragma once
struct MemoryState {
 CompactState storage;const int* order=nullptr;const int* inverse=nullptr;
 const int* laneMap=nullptr;const int* best=nullptr;
 const MotionCar* cache=nullptr;const int* cacheInverse=nullptr;int cacheBase=0,cacheCount=0;
 __device__ MotionCar load(int id) const {
  if(cache){int s=cacheInverse[id]-cacheBase;if(s>=0&&s<cacheCount)return cache[s];}
  return storage.load(inverse?inverse[id]:id);
 }
 __device__ void store(int id,const MotionCar& c) const {storage.store(inverse?inverse[id]:id,c);}
};
__global__ void inverseSlots(const int* order,int* inverse,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)inverse[order[i]]=i;}
__global__ void extractMemory(const Car* cars,MemoryState dst){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=dst.storage.count)return;const Car& a=cars[i];MotionCar c;
c.state=a.state;
c.k=a.k;
c.n=a.n;
c.mid=a.mid;
c.event=a.event;
c.denial=a.denial;
c.departure=a.departure;
c.pos=a.pos;
c.v=a.v;
c.stopped=a.stopped;
c.entryWait=a.entryWait;
c.edge=a.edges[a.k];c.lane=a.lanes[a.k];c.movement=a.moves[a.k];dst.store(i,c);}

__global__ void buildNearestConnector(const Car* cars,const int* head,const int* links,const int* laneKeys,int* best,int count){
 int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot>=count)return;int key=laneKeys[slot],chosen=-1;float pos=1.e30f;
 for(int j=head[key];j>=0;j=links[j]){float p=cars[j].pos;if(p<pos||(p==pos&&(chosen<0||j<chosen))){pos=p;chosen=j;}}
 best[slot]=chosen;
}
__global__ void advanceMemory(MemoryState old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive,const int* leaders=nullptr,const int* work=nullptr,int region=-1,const int* connectorHead=nullptr,const int* connectorNext=nullptr,const int* movementOpen=nullptr){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;if(old.order)i=old.order[i];if(work&&work[i]!=region){writeCompact(next[i],old.load(i));return;}MotionCar c=old.load(i);c.event=0;c.denial=0;
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
__global__ void advanceShared(MemoryState old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive,const int* leaders=nullptr,const int* work=nullptr,int region=-1,const int* connectorHead=nullptr,const int* connectorNext=nullptr,const int* movementOpen=nullptr){
 __shared__ MotionCar local[128];int slot=blockIdx.x*blockDim.x+threadIdx.x;
 int i=slot<n?old.order[slot]:0;if(slot<n)local[threadIdx.x]=old.load(i);__syncthreads();if(slot>=n)return;
 old.cache=local;old.cacheBase=blockIdx.x*128;old.cacheCount=min(128,n-old.cacheBase);if(work&&work[i]!=region){writeCompact(next[i],old.load(i));return;}MotionCar c=old.load(i);c.event=0;c.denial=0;
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
