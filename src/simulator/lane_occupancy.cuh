#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

// Byte CAS must retry when another byte in the same word changes. Only a
// successful comparison of the entire word establishes ownership of the byte.
__device__ inline unsigned int atomicCASUchar(unsigned char* address,
                                             unsigned char expected,
                                             unsigned char value) {
  auto* word = reinterpret_cast<unsigned int*>(reinterpret_cast<uintptr_t>(address) & ~uintptr_t(3));
  const unsigned shift = (reinterpret_cast<uintptr_t>(address) & 3) * 8;
  const unsigned mask = 0xFFu << shift;
  unsigned observed = atomicCAS(word, 0u, 0u);
  for (;;) {
    if (((observed >> shift) & 0xFFu) != expected) return 0;
    const unsigned desired = (observed & ~mask) | (unsigned(value) << shift);
    const unsigned actual = atomicCAS(word, observed, desired);
    if (actual == observed) return 1;
    observed = actual;
  }
}

// Keep every old occupied cell available to its owner for the whole timestep.
// A follower can use a vacated cell next step, never before its leader commits.
// ownWriteCell is UINT_MAX for a newly inserted vehicle.
__device__ inline bool reserveLaneCell(unsigned char* map, unsigned readShift,
                                      unsigned writeShift, unsigned target,
                                      unsigned ownWriteCell, unsigned char speed) {
  if (target != ownWriteCell && map[readShift + target - writeShift] != 0xFF)
    return false;
  return atomicCASUchar(map + target, 0xFF, speed) != 0;
}
