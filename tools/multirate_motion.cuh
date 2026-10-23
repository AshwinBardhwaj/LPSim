// Generated from compact baseline. Approximate held acceleration, guarded every tick.
__global__ void advanceMulti(CompactState old,Car* next,int n,const Road* roads,
                        const Movement* moves,const int* conflict,int nm,float t,float dt,int mode,bool exclusive,const int* leaders,const int* work,int region,const int* connectorHead,const int* connectorNext,const int* movementOpen,float* held,int* previousEdge,int step,int rate,unsigned long long* counts){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;if(work&&work[i]!=region){writeCompact(next[i],old.load(i));return;}MotionCar c=old.load(i);c.event=0;c.denial=0;
 if(counts&&c.state==1)atomicAdd(counts+1,1ULL);
 if(c.state!=1)previousEdge[i]=-1;
 if(c.state==1){const int edge=c.edge,lane=c.lane;const Road r=roads[edge];
  float horizon=rate*dt;
  bool eligible=leaders[i]<0&&connectorHead[edge*8+lane]<0&&r.length-c.pos>fmaxf(c.v,r.speed)*horizon+SPACING+10.f;
  bool reuse=eligible&&previousEdge[i]==edge&&step%rate!=0;
  float acc,bound=r.length;
  if(reuse){acc=held[i];if(counts)atomicAdd(counts,1ULL);}
  else {
  bool open=c.k==c.n-1;
  if(c.k+1<c.n){int mid=c.movement;const Movement m=moves[mid];open=indication(m,t,mode)==2;
   if(movementOpen)open=movementOpen[mid];
   else if(open)for(int j=0;j<n;++j){const MotionCar d=old.load(j);
    if((d.state==2&&conflictsWith(conflict,nm,mid,d.mid)&&(exclusive||mid!=d.mid))||
       (d.state==1&&d.edge==m.out&&d.lane==m.outLane&&d.pos<SPACING+8.f)){open=false;break;}}
  }
  float gap=open?1.e6f:r.length-c.pos+2,leaderV=0;
  if(leaders){if(leaders[i]>=0){const MotionCar d=old.load(leaders[i]);float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}else for(int j=0;j<n;++j){if(j==i)continue;const MotionCar d=old.load(j);if(d.state==1&&d.edge==edge&&d.lane==lane&&d.pos>c.pos){float g=d.pos-c.pos-4.5f;if(g<gap){gap=g;leaderV=d.v;}bound=fminf(bound,d.pos-SPACING);}}
  int chosenConnector=-1;
  for(int j=connectorHead?connectorHead[edge*8+lane]:0;j>=0&&j<n;j=connectorHead?connectorNext[j]:j+1){const MotionCar d=old.load(j);if(d.state==2&&moves[d.mid].in==edge&&moves[d.mid].inLane==lane){
   float distance=r.length-c.pos+d.pos;float g=distance-4.5f;
   // Preserve the original ascending-ID tie choice despite atomic list order.
   if(g<gap||(g==gap&&chosenConnector>=0&&j<chosenConnector)){gap=g;leaderV=d.v;chosenConnector=j;}
   bound=fminf(bound,r.length+d.pos-SPACING);
  }}
  float desired=1.38075f+fmaxf(0.f,c.v*.5433028f+c.v*(c.v-leaderV)/(2*sqrtf(.5570409f*2.902058f)));
  acc=.5570409f*(1-powf(c.v/r.speed,4)-powf(desired/fmaxf(.05f,gap),2));
  held[i]=acc;
  }
  previousEdge[i]=eligible?edge:-1;
  float speed=fmaxf(0.f,c.v+acc*dt);float proposed=c.pos+.5f*(c.v+speed)*dt;
  if(proposed>bound){proposed=fmaxf(c.pos,bound);speed=0;}
  c.pos=proposed;c.v=speed;
  if(r.length-c.pos<.1f && c.v<.3f){c.pos=r.length;c.v=0;}
  c.stopped=(c.v<.1f&&r.length-c.pos<.11f)?c.stopped+dt:0;
  if(c.k==c.n-1&&r.length-c.pos<.11f)c.state=3;
 }
 writeCompact(next[i],c);
}
