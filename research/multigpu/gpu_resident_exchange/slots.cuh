#pragma once
#include "protocol.h"
#include <climits>

namespace resident {
struct Message { int id, owner; Car car; };
struct Slots {
    int capacity, total, rank, ranks;
    int *ids, *lookup, *owners, *freeSlots, *freeCount, *error;
    unsigned long long* refreshed;
};
__device__ inline void fail(int* error, Error value) { atomicCAS(error, 0, int(value)); }

__global__ void resetOutboxes(Envelope* heads, int ranks, int rank,
                             unsigned long long epoch, int phase, const int* error) {
    int d = threadIdx.x;
    if (d < ranks) heads[d] = {epoch, phase, rank, d, 0, *error};
}
// Classification is fused with packing. No separate membership-detection pass.
__global__ void packSlotsState(Car* cars, Slots slots, const Road* roads,
                              const Movement* moves, const int* jo,
                              Message* out, Envelope* heads, int phase) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots.capacity || slots.ids[i] < 0 || *slots.error ||
        slots.owners[i] != slots.rank) return;
    Car c = cars[i];
    int owner = authority(c, roads, moves, jo);
    int ghost = c.state == 1 ? jo[roads[c.edges[c.k]].from] : owner;
    if (phase && owner != slots.rank) { fail(slots.error, WrongMigration); return; }
    int destinations[2] = {owner, ghost};
    for (int k = 0; k < 2; ++k) {
        int d = destinations[k];
        if (d == slots.rank || (k && ghost == owner)) continue;
        int index = atomicAdd(&heads[d].count, 1);
        if (index >= slots.capacity) { fail(slots.error, SendOverflow); continue; }
        out[d * slots.capacity + index] = {slots.ids[i], owner, c};
    }
    // Existing owner revokes its authority after capturing the outgoing state.
    slots.owners[i] = owner;
    if (owner == slots.rank || ghost == slots.rank)
        slots.refreshed[i] = heads[0].epoch;
}
__global__ void finalizeOutboxes(Envelope* heads, int ranks, const int* error) {
    if (threadIdx.x < ranks) heads[threadIdx.x].error = *error;
}
// Run after incoming refreshes: unchanged ghosts retain their slot. Only obsolete
// ghosts/migrated owners are reclaimed. Capacity checks charge transient slots.
__global__ void retireSlots(Car* cars, Slots slots, unsigned long long epoch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots.capacity || slots.ids[i] < 0 || *slots.error) return;
    if (slots.owners[i] == slots.rank || slots.refreshed[i] == epoch) return;
    slots.lookup[slots.ids[i]] = -1;
    slots.ids[i] = -1;
    cars[i] = Car{};
    cars[i].state = -1;
    slots.freeSlots[atomicAdd(slots.freeCount, 1)] = i;
}
// One invocation per source, ordered on the destination stream. IDs within one
// source batch are unique by construction; repeated refreshes are rejected.
// remoteHeads/remoteMessages may be peer memory. CUDA event dependencies establish
// publication visibility and consumption completion; no spin loop is required.
__global__ void pullState(const Envelope* remoteHeads, const Message* remoteMessages,
                          int sourceCapacity, Car* cars, Slots slots, int source,
                          unsigned long long epoch, int phase) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    const Envelope h = remoteHeads[slots.rank];
    if (h.error) { if (!k) atomicCAS(slots.error, 0, h.error); return; }
    if (h.epoch != epoch || h.phase != phase || h.source != source ||
        h.destination != slots.rank || h.count < 0 || h.count > sourceCapacity) {
        if (!k) fail(slots.error, BadEnvelope);
        return;
    }
    if (k >= h.count || *slots.error) return;
    const Message m = remoteMessages[slots.rank * sourceCapacity + k];
    if (m.id < 0 || m.id >= slots.total || m.owner < 0 || m.owner >= slots.ranks ||
        (phase && m.owner != source)) { fail(slots.error, BadOwner); return; }
    int slot = atomicCAS(slots.lookup + m.id, -1, -2);
    if (slot == -2) { fail(slots.error, Duplicate); return; }
    if (slot < 0) {
        int top = atomicSub(slots.freeCount, 1) - 1;
        if (top < 0) { fail(slots.error, ReceiveOverflow); return; }
        slot = slots.freeSlots[top];
        atomicExch(slots.lookup + m.id, slot);
        slots.ids[slot] = m.id;
    }
    if (atomicExch(slots.refreshed + slot, epoch) == epoch) { fail(slots.error, Duplicate); return; }
    cars[slot] = m.car;
    slots.owners[slot] = m.owner;
}

struct StableSlotLess {
    const int* ids;
    __device__ bool operator()(int a, int b) const {
        int x = ids[a] < 0 ? INT_MAX : ids[a], y = ids[b] < 0 ? INT_MAX : ids[b];
        return x != y ? x < y : a < b;
    }
};
__global__ void gatherSlotView(const Car* cars, Car* out, Slots slots, const int* order,
                               int* outIDs, int* outOwners, unsigned long long* outEpochs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots.capacity || *slots.error) return;
    int src = order[i]; out[i] = cars[src]; outIDs[i] = slots.ids[src];
    outOwners[i] = slots.owners[src]; outEpochs[i] = slots.refreshed[src];
}
__global__ void installSlotView(Car* cars, const Car* gathered, Slots slots,
                                const int* ids, const int* owners, const unsigned long long* epochs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots.capacity || *slots.error) return;
    cars[i] = gathered[i]; slots.ids[i] = ids[i]; slots.owners[i] = owners[i]; slots.refreshed[i] = epochs[i];
    if (ids[i] >= 0) slots.lookup[ids[i]] = i;
    else slots.freeSlots[atomicAdd(slots.freeCount, 1)] = i;
}
__global__ void resetSpawnIDs(JunctionSummary* summary) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < summary->capacity) summary->spawn[i] = INT_MAX;
}
__global__ void makeSlotRequestKeys(const EntryRequest* req, const Movement* moves,
                                   const int* ids, unsigned long long* keys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    EntryRequest r = req[i];
    keys[i] = r.pass < 0 ? ~0ULL : (static_cast<unsigned long long>(moves[r.mid].junction) << 33)
        | (static_cast<unsigned long long>(r.pass) << 32) | unsigned(ids[i]);
}
__global__ void spawnSlots(Car* cars, const Slots slots, int nr, const JunctionSummary* s,
                           const int* selected = nullptr, int count = 0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (selected ? count : nr * 8)) return;
    int lane = selected ? selected[index] : index;
    int id = s->spawn[lane];
    if (id < slots.total && s->nearest[lane] >= SPACING && !s->reserved[lane]) {
        int slot = slots.lookup[id];
        if (slot < 0 || slots.owners[slot] != slots.rank) { fail(slots.error, BadOwner); return; }
        cars[slot].state = 1; cars[slot].pos = 0; cars[slot].v = 0;
    }
}
// Comparator preserves the frozen sorted-global-ID leader order, regardless of
// physical slot reuse. This is a charged ablation versus the baseline radix path.
struct SlotLaneLess {
    const Car* cars;
    const int* ids;
    __device__ bool operator()(int a, int b) const {
        const Car& x = cars[a]; const Car& y = cars[b];
        bool xa = x.state == 1, ya = y.state == 1;
        if (xa != ya) return xa;
        if (xa) {
            if (x.edges[x.k] != y.edges[y.k]) return x.edges[x.k] < y.edges[y.k];
            if (x.lanes[x.k] != y.lanes[y.k]) return x.lanes[x.k] < y.lanes[y.k];
            if (x.pos != y.pos) return x.pos < y.pos;
        }
        return ids[a] != ids[b] ? ids[a] < ids[b] : a < b;
    }
};
}
