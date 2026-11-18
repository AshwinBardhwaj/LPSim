#pragma once
#include <array>
// Separate diagnostic binary only. No events/rows in the timing binary.
struct ExchangeTrace {
 int rank,step=0; const char* stage="unset";
#ifdef EXCHANGE_DIAGNOSTICS
 struct Span {cudaEvent_t a,b;const char* label;long long begin,end;};
 struct Row {int step;std::string stage,label;float gpu;long long begin,end;};
 std::array<Span,16> spans;int used=0;bool initialized=false;
 std::vector<Row> rows;
 static long long now(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
#endif
 explicit ExchangeTrace(int r):rank(r){}
 void reset(const char* s){stage=s;
#ifdef EXCHANGE_DIAGNOSTICS
 if(!initialized){for(auto& x:spans){check(cudaEventCreate(&x.a));check(cudaEventCreate(&x.b));}initialized=true;}used=0;
#endif
 }
 template<class F> void part(const char* label,F fn){
#ifdef EXCHANGE_DIAGNOSTICS
 if(used>=int(spans.size()))throw std::runtime_error("Trace capacity");
 auto& x=spans[used++];x.label=label;x.begin=now();check(cudaEventRecord(x.a));fn();check(cudaEventRecord(x.b));x.end=now();
#else
 fn();
#endif
 }
 void sync(){
#ifdef EXCHANGE_DIAGNOSTICS
 auto begin=now();check(cudaDeviceSynchronize());auto end=now();
 rows.push_back({step,stage,"host_device_sync",-1,begin,end});
 for(int i=0;i<used;i++){float ms;check(cudaEventElapsedTime(&ms,spans[i].a,spans[i].b));rows.push_back({step,stage,spans[i].label,ms,spans[i].begin,spans[i].end});}used=0;
#else
 check(cudaDeviceSynchronize());
#endif
 }
 void write(std::ofstream& f){
#ifdef EXCHANGE_DIAGNOSTICS
 for(auto& x:rows)f<<x.step<<','<<rank<<','<<x.stage<<','<<x.label<<','<<x.gpu<<','<<x.begin<<','<<x.end<<'\n';
#endif
 }
};
