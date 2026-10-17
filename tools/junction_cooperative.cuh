// Experimental warp-per-junction arbitration. Priority and state commits remain
// serial; independent conflict queries against prior grants run across a warp.
__global__ void arbitrateCooperativeJunctions(Car* cars,const Road* roads,const Movement* moves,const int* conflict,int nm,int nj,bool exclusive,JunctionSummary* s,const EntryRequest* requests,const unsigned long long* keys,const int* begin,const int* end,int threshold){
 int lane=threadIdx.x&31,junction=(blockIdx.x*blockDim.x+threadIdx.x)/32;if(junction>=nj)return;
 int first=begin[junction],last=end[junction];if(last-first<threshold||first==last)return;
 int* granted=s->granted+s->junctionOffsets[junction];int count=0;
 for(int k=first;k<last;++k){
  int i=static_cast<unsigned int>(keys[k]);EntryRequest r=requests[i];Movement m=moves[r.mid];int key=m.out*8+m.outLane;float speed=m.turn==0?7.f:4.f;
  int eligible=0;
  if(lane==0){Car& c=cars[i];if(c.state==1){
   if(s->reserved[key]&&(exclusive||s->owner[key]!=r.mid))c.denial=1;
   else if(!exclusive&&s->tail[r.mid]<fmaxf(SPACING,2.f*speed))c.denial=1;
   else if(fminf(s->nearest[key],roads[m.out].length)<SPACING*(s->reserved[key]+1)+8.f)c.denial=2;
   else eligible=1;
  }}
  eligible=__shfl_sync(0xffffffff,eligible,0);
  if(eligible){
   int hit=nm;
   for(int q=lane;q<count;q+=32){int j=granted[q];if(j<=r.blockedIndex&&(exclusive||j!=r.mid)&&conflictsWith(conflict,nm,r.mid,j))hit=min(hit,j);}
   for(int offset=16;offset;offset/=2)hit=min(hit,__shfl_down_sync(0xffffffff,hit,offset));
   int admitted=0;
   if(lane==0){Car& c=cars[i];int blocked=r.blockedIndex,reason=r.blockedReason;if(hit<nm&&hit<=blocked){blocked=hit;reason=1;}
    if(blocked<nm)c.denial=reason;
    else {c.entryWait=c.stopped;c.state=2;c.mid=r.mid;c.pos=0;c.v=speed;c.event=r.pass==1?2:1;c.stopped=0;s->tail[r.mid]=0;s->reserved[key]++;s->owner[key]=r.mid;granted[count]=r.mid;admitted=1;}
   }
   admitted=__shfl_sync(0xffffffff,admitted,0);count+=admitted;
  }
  __syncwarp();
 }
}
__global__ void arbitrateSmallJunctions(Car* cars,int n,const Road* roads,const Movement* moves,const int* conflict,int nm,int nj,bool exclusive,JunctionSummary* s,const EntryRequest* requests,const unsigned long long* keys,const int* begin,const int* end,int threshold){
 int junction=blockIdx.x*blockDim.x+threadIdx.x;if(junction>=nj||end[junction]-begin[junction]>=threshold)return;
 int* granted=s->granted+s->junctionOffsets[junction];int count=0;
 for(int k=begin[junction];k<end[junction];++k){int i=static_cast<unsigned int>(keys[k]);EntryRequest r=requests[i];admitJunctionRequest(cars,i,r.pass,roads,moves,conflict,nm,exclusive,s,r,granted,count);}
}
