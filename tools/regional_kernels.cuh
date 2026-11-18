#pragma once
// Generated exact subset kernels and certified road-local temporal motion.

__global__ void prepareJunctionsSubset(Car* cars,int n,const Movement* moves,float dt,JunctionSummary* s,const int* ids,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<s->capacity){s->nearest[i]=s->arrival[i]=s->tail[i]=1.e9f;s->busy[i]=s->reserved[i]=0;s->owner[i]=-1;s->spawn[i]=n;}
 if(i<count){Car& c=cars[ids[i]];if(c.state==2){const Movement m=moves[c.mid];c.pos+=c.v*dt;
  if(c.pos>=m.length){c.pos-=m.length;c.k++;c.state=1;c.event=3;c.stopped=0;c.mid=-1;}}}
}

__global__ void summarizeJunctionsSubset(const Car* cars,int n,const Road* roads,const Movement* moves,float t,int mode,JunctionSummary* s,const int* ids,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;i=ids[i];const Car& c=cars[i];
 if(c.state==2){int key=moves[c.mid].out*8+moves[c.mid].outLane;atomicAdd(s->busy+c.mid,1);minPositive(s->tail+c.mid,c.pos);atomicAdd(s->reserved+key,1);atomicMax(s->owner+key,c.mid);}
 if(c.state==1){minPositive(s->nearest+c.edges[c.k]*8+c.lanes[c.k],c.pos);
  if(c.k+1<c.n){int mid=c.moves[c.k];if(indication(moves[mid],t,mode)==2)minPositive(s->arrival+mid,(roads[c.edges[c.k]].length-c.pos)/fmaxf(1.f,c.v));}}
 if(c.state==0&&c.departure<=t)atomicMin(s->spawn+c.edges[0]*8+c.lanes[0],i);
}

__global__ void requestJunctionEntrySubset(const Car* cars,int n,const Road* roads,const Movement* moves,const int* conflict,int nm,float t,int mode,bool exclusive,const JunctionSummary* s,EntryRequest* requests,const int* ids,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;i=ids[i];const Car& c=cars[i];EntryRequest r{-1,-1,nm,0};
 if(c.state==1&&c.k+1<c.n&&roads[c.edges[c.k]].length-c.pos<=.11f){
  int mid=c.moves[c.k];const Movement m=moves[mid];int light=indication(m,t,mode);
  bool red=light==0&&m.turn==-1&&m.rtor&&c.stopped>=2.f&&!clearance(t,mode);
  if(light==2||red){r.mid=mid;r.pass=light==2?0:1;
   for(int ci=conflict[mid];ci<conflict[mid+1];++ci){int j=conflict[nm+1+ci];
    if(s->busy[j]&&(exclusive||j!=mid)){r.blockedIndex=j;r.blockedReason=1;break;}
    if(r.pass==1&&j!=mid&&s->arrival[j]<4.f){r.blockedIndex=j;r.blockedReason=3;break;}
   }
  }
 }
 requests[i]=r;
}

__global__ void extractCompactSubset(const Car* cars, CompactState out,const int* ids,int count) { int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;i=ids[i];const Car& a=cars[i];MotionCar c;
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
 c.edge=a.edges[a.k];c.lane=a.lanes[a.k];c.movement=a.moves[a.k];out.store(i,c);}

__global__ void advanceCompactSubset(CompactState old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive,const int* leaders=nullptr,const int* work=nullptr,int region=-1,const int* connectorHead=nullptr,const int* connectorNext=nullptr,const int* movementOpen=nullptr,const int* ids=nullptr,int count=0){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;i=ids[i];if(work&&work[i]!=region){writeCompact(next[i],old.load(i));return;}MotionCar c=old.load(i);c.event=0;c.denial=0;
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
  int chosenConnector=-1;
  for(int j=connectorHead?connectorHead[edge*8+lane]:0;j>=0&&j<n;j=connectorHead?connectorNext[j]:j+1){const MotionCar d=old.load(j);if(d.state==2&&moves[d.mid].in==edge&&moves[d.mid].inLane==lane){
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

__global__ void collectMotionContextSubset(const Car* cars,int n,const Movement* moves,
 int* head,int* links,int* blocked,int* busy,int* queried,const int* ids,int count){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;i=ids[i];const Car& c=cars[i];
 if(c.state==2){const Movement m=moves[c.mid];if(head)links[i]=atomicExch(head+m.in*8+m.inLane,i);if(busy)atomicExch(busy+c.mid,1);}
 if(blocked&&c.state==1){if(c.pos<SPACING+8.f)atomicExch(blocked+c.edges[c.k]*8+c.lanes[c.k],1);if(c.k+1<c.n)atomicExch(queried+c.moves[c.k],1);}
}


__global__ void makeSubsetLeaderKeys(const Car* cars,const int* work,int count,unsigned long long* keys,int* ids){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=work[k];ids[k]=i;const Car& c=cars[i];if(c.state!=1){keys[k]=~0ULL;return;}unsigned int bits=__float_as_uint(c.pos==0.f?0.f:c.pos),ordered=bits&0x80000000u?~bits:bits^0x80000000u;unsigned int lane=c.edges[c.k]*8+c.lanes[c.k];keys[k]=(static_cast<unsigned long long>(lane)<<32)|ordered;}
void buildSubsetLeaders(LeaderIndex& x,const Car* cars,const int* work,int count){if(!count)return;makeSubsetLeaderKeys<<<(count+127)/128,128>>>(cars,work,count,x.keysIn,x.ids);check(cub::DeviceRadixSort::SortPairs(x.temporary,x.temporaryBytes,x.keysIn,x.keysOut,x.ids,x.order,count));linkLaneLeaders<<<(count+127)/128,128>>>(cars,x.order,x.leaders,count);}
__global__ void makeSubsetRequests(const EntryRequest* requests,const Movement* moves,const int* ids,int count,unsigned long long* keys){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;int i=ids[k];EntryRequest q=requests[i];keys[k]=q.pass<0?~0ULL:(static_cast<unsigned long long>(moves[q.mid].junction)<<33)|(static_cast<unsigned long long>(q.pass)<<32)|static_cast<unsigned int>(i);}
__global__ void publishLocal(Car* cars,const MotionCar* cache,const int* ids,int count,int h){int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<count)writeCompact(cars[ids[k]],cache[h*count+k]);}


template<int Width> __global__ void localRoadSteps(const Car* cars,const Road* roads,const int* offsets,const int* ids,int selected,int steps,float dt,MotionCar* cache,int groupStart,int groupEnd,bool trajectory=true){
 __shared__ float first[14*128],second[14*128];int localGroup=threadIdx.x/Width,group=groupStart+blockIdx.x*(128/Width)+localGroup,offset=group<groupEnd?offsets[group]:0,count=group<groupEnd?offsets[group+1]-offset:0,i=threadIdx.x%Width;CompactState old{first+localGroup*14*Width,count},next{second+localGroup*14*Width,count};
 if(i<count){const Car& z=cars[ids[offset+i]];MotionCar c;c.state=z.state;c.k=z.k;c.n=z.n;c.mid=z.mid;c.event=z.event;c.denial=z.denial;c.departure=z.departure;c.pos=z.pos;c.v=z.v;c.stopped=z.stopped;c.entryWait=z.entryWait;c.edge=z.edges[z.k];c.lane=z.lanes[z.k];c.movement=z.moves[z.k];old.store(i,c);}__syncthreads();
 for(int h=0;h<steps;++h){if(i<count){MotionCar c=old.load(i);c.event=0;c.denial=0;
 if(c.state==1){const int edge=c.edge,lane=c.lane;const Road r=roads[edge];
  bool open=c.k==c.n-1;
  float gap=open?1.e6f:r.length-c.pos+2,leaderV=0,bound=r.length;
  int lead=i+1<old.count&&old.load(i+1).lane==lane?i+1:-1;
  if(lead>=0){MotionCar d=old.load(lead);float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}
  float desired=1.38075f+fmaxf(0.f,c.v*.5433028f+c.v*(c.v-leaderV)/(2*sqrtf(.5570409f*2.902058f)));
  float acc=.5570409f*(1-powf(c.v/r.speed,4)-powf(desired/fmaxf(.05f,gap),2));
  float speed=fmaxf(0.f,c.v+acc*dt);float proposed=c.pos+.5f*(c.v+speed)*dt;
  if(proposed>bound){proposed=fmaxf(c.pos,bound);speed=0;}
  c.pos=proposed;c.v=speed;
  if(r.length-c.pos<.1f && c.v<.3f){c.pos=r.length;c.v=0;}
  c.stopped=(c.v<.1f&&r.length-c.pos<.11f)?c.stopped+dt:0;
  if(c.k==c.n-1&&r.length-c.pos<.11f)c.state=3;
 }
next.store(i,c);if(trajectory||h==steps-1)cache[h*selected+offset+i]=c;}__syncthreads();float* tmp=old.values;old.values=next.values;next.values=tmp;}
}
