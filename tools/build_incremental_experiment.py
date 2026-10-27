from pathlib import Path
r=Path(__file__).resolve().parents[1];s=(r/'tools/gating_multirate_simulator.cu').read_text()
s=s.replace('#include "junction_request_index.cuh"','#include "junction_request_index.cuh"\n#include "incremental_leaders.cuh"')
s=s.replace(' int steps=int(std::round(duration/dt));',''' const std::string indexPolicy=std::getenv("LPSIM_INDEX_POLICY")?std::getenv("LPSIM_INDEX_POLICY"):"bulk";
 bool indexTiming=std::getenv("LPSIM_TIMING_ONLY");IncrementalLeaders incremental(n,nr,!indexTiming);LeaderIndex referenceIndex(n,true,true);
 std::ofstream indexMetrics("index_metrics.csv");indexMetrics<<"time,index_ms,reference_bulk_ms,membership_changes,dirty_lanes\\n";
 cudaEvent_t indexEvents[3];for(auto& e:indexEvents)check(cudaEventCreate(&e));
 std::vector<int> observedLeaders(n),expectedLeaders(n);
 int steps=int(std::round(duration/dt));''')
s=s.replace('check(cudaEventRecord(ev0));laneIndex.build(a);','''check(cudaEventRecord(ev0));check(cudaEventRecord(indexEvents[0]));
 if(indexPolicy=="bulk"){laneIndex.build(a);leaders=laneIndex.leaders;}else{incremental.build(a,indexPolicy=="periodic8"&&step%8==0);leaders=incremental.bulk.leaders;}
 check(cudaEventRecord(indexEvents[1]));
 if(!indexTiming){referenceIndex.build(a);check(cudaEventRecord(indexEvents[2]));check(cudaEventSynchronize(indexEvents[2]));float own,ref;check(cudaEventElapsedTime(&own,indexEvents[0],indexEvents[1]));check(cudaEventElapsedTime(&ref,indexEvents[1],indexEvents[2]));
 check(cudaMemcpy(observedLeaders.data(),leaders,n*4,cudaMemcpyDeviceToHost));check(cudaMemcpy(expectedLeaders.data(),referenceIndex.leaders,n*4,cudaMemcpyDeviceToHost));if(observedLeaders!=expectedLeaders)throw std::runtime_error("Incremental leader mismatch at step "+std::to_string(step));
 unsigned int counts[2]={0,0};if(indexPolicy!="bulk")check(cudaMemcpy(counts,incremental.stats,8,cudaMemcpyDeviceToHost));indexMetrics<<t<<','<<own<<','<<ref<<','<<counts[0]<<','<<counts[1]<<'\\n';}
''')
(r/'tools/incremental_simulator.cu').write_text('// Generated index-maintenance experiment; production unchanged.\n'+s)
