#include <cstdio>
#include <cstdlib>
#include "../src/simulator/lane_occupancy.cuh"
#define CUDA_OK(call) do { auto e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s\n", cudaGetErrorString(e)); exit(2); } } while (0)

__global__ void contention(unsigned* failures) {
  __shared__ unsigned word;
  __shared__ unsigned winners[4];
  const unsigned lane = threadIdx.x % 4;
  for (int repetition = 0; repetition < 10000; ++repetition) {
    if (threadIdx.x == 0) word = 0xFFFFFFFF;
    if (threadIdx.x < 4) winners[threadIdx.x] = 0;
    __syncthreads();
    if (atomicCASUchar(reinterpret_cast<unsigned char*>(&word) + lane, 0xFF, 17 + lane))
      atomicAdd(winners + lane, 1);
    __syncthreads();
    if (threadIdx.x < 4 && (winners[lane] != 1 ||
        ((word >> (lane * 8)) & 0xFF) != 17 + lane)) atomicAdd(failures, 1);
    __syncthreads();
  }
}

__global__ void rollbackProtection(unsigned* failures) {
  __shared__ unsigned storage[2];
  auto* map = reinterpret_cast<unsigned char*>(storage);
  // Previous frame: car A at cell 0, car B at cell 1. Next frame empty.
  if (threadIdx.x == 0) { storage[0] = 0xFFFF0000; storage[1] = 0xFFFFFFFF; }
  __syncthreads();
  if (threadIdx.x == 0) {
    // A cannot take B's old position, even before B publishes its rollback.
    if (reserveLaneCell(map, 0, 4, 5, 4, 9)) atomicAdd(failures, 1);
    if (!reserveLaneCell(map, 0, 4, 4, 4, 0)) atomicAdd(failures, 1);
  }
  if (threadIdx.x == 1) {
    if (!reserveLaneCell(map, 0, 4, 5, 5, 0)) atomicAdd(failures, 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (map[4] != 0 || map[5] != 0) atomicAdd(failures, 1);
    // An entrant cannot claim a protected old cell or an already-reserved new cell.
    if (reserveLaneCell(map, 0, 4, 4, ~0u, 3)) atomicAdd(failures, 1);
    if (!reserveLaneCell(map, 0, 4, 6, ~0u, 3)) atomicAdd(failures, 1);
    if (reserveLaneCell(map, 0, 4, 6, ~0u, 4)) atomicAdd(failures, 1);
  }
}
int main() {
  unsigned* failures; CUDA_OK(cudaMallocManaged(&failures, sizeof(unsigned)));
  *failures = 0;
  contention<<<64, 128>>>(failures);
  rollbackProtection<<<64, 32>>>(failures);
  CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
  const auto count = *failures; CUDA_OK(cudaFree(failures));
  printf("Adjacent-byte contention / rollback protection failures: %u\n", count);
  return count ? 1 : 0;
}
