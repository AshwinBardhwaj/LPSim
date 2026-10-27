#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <chrono>
void ok(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
__global__ void counterProbe(float* a){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<4096)a[i]=a[i]*1.0001f+1;}
int main(int argc,char** argv){try{int n;ok(cudaGetDeviceCount(&n));if(n<1)throw std::runtime_error("No CUDA devices");
 if(argc>1){ok(cudaSetDevice(0));float* x;ok(cudaMalloc(&x,4096*4));ok(cudaMemset(x,0,4096*4));counterProbe<<<32,128>>>(x);ok(cudaDeviceSynchronize());ok(cudaFree(x));return 0;}
 std::cout<<"{\"devices\":[";for(int i=0;i<n;++i){cudaDeviceProp p;ok(cudaGetDeviceProperties(&p,i));char bus[32];ok(cudaDeviceGetPCIBusId(bus,32,i));if(i)std::cout<<',';std::cout<<"{\"id\":"<<i<<",\"name\":\""<<p.name<<"\",\"pci_bus\":\""<<bus<<"\",\"memory_bytes\":"<<p.totalGlobalMem<<",\"sm_count\":"<<p.multiProcessorCount<<",\"compute_major\":"<<p.major<<",\"compute_minor\":"<<p.minor<<'}';}
 std::cout<<"],\"transfers\":[";bool first=true;
 for(int src=0;src<n;++src)for(int dst=0;dst<n;++dst)if(src!=dst){int peer;ok(cudaDeviceCanAccessPeer(&peer,dst,src));ok(cudaSetDevice(dst));if(peer){auto e=cudaDeviceEnablePeerAccess(src,0);if(e!=cudaErrorPeerAccessAlreadyEnabled)ok(e);else cudaGetLastError();}
 for(size_t bytes:{size_t(64),size_t(65536),size_t(1048576)}){void *a,*b;ok(cudaSetDevice(src));ok(cudaMalloc(&a,bytes));ok(cudaMemset(a,0,bytes));ok(cudaDeviceSynchronize());ok(cudaSetDevice(dst));ok(cudaMalloc(&b,bytes));cudaStream_t stream;ok(cudaStreamCreate(&stream));cudaEvent_t start,end;ok(cudaEventCreate(&start));ok(cudaEventCreate(&end));for(int k=0;k<10;++k)ok(cudaMemcpyPeerAsync(b,dst,a,src,bytes,stream));ok(cudaStreamSynchronize(stream));auto wall=std::chrono::steady_clock::now();ok(cudaEventRecord(start,stream));for(int k=0;k<100;++k)ok(cudaMemcpyPeerAsync(b,dst,a,src,bytes,stream));ok(cudaEventRecord(end,stream));ok(cudaEventSynchronize(end));float ms;ok(cudaEventElapsedTime(&ms,start,end));double wallUs=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-wall).count()/100;
 if(!first)std::cout<<',';first=false;std::cout<<"{\"source\":"<<src<<",\"destination\":"<<dst<<",\"peer_access\":"<<(peer?"true":"false")<<",\"bytes\":"<<bytes<<",\"batch_average_device_us\":"<<ms*10<<",\"batch_average_wall_us\":"<<wallUs<<'}';ok(cudaEventDestroy(start));ok(cudaEventDestroy(end));ok(cudaStreamDestroy(stream));ok(cudaFree(b));ok(cudaSetDevice(src));ok(cudaFree(a));}}
 std::cout<<"]}\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
