// Included after Car, check(), and the comparator leader-index kernels.
// Finite positions, road IDs < 2^29, and lane IDs 0..7 are the key contract.
// Stable CUB sorting preserves ascending vehicle IDs for equal keys.
__global__ void makeLeaderKeys(const Car* cars, unsigned long long* keys,
                               int* ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ids[i] = i;
    const Car& c = cars[i];
    if (c.state != 1) {
        keys[i] = ~0ULL; // Inactive records follow all road-active records.
        return;
    }
    // Match float comparisons, including equality of -0 and +0.
    const unsigned int bits = __float_as_uint(c.pos == 0.f ? 0.f : c.pos);
    const unsigned int ordered = bits & 0x80000000u ? ~bits : bits ^ 0x80000000u;
    const unsigned int lane = static_cast<unsigned int>(c.edges[c.k]) * 8u
                            + static_cast<unsigned int>(c.lanes[c.k]);
    keys[i] = (static_cast<unsigned long long>(lane) << 32) | ordered;
}

// Own persistent buffers; rebuild from the immutable pre-motion state each step.
// Sorting IDs does not reorder Car storage or change junction priority IDs.
struct LeaderIndex {
    int n;
    bool enabled, radix;
    int *order = nullptr, *leaders = nullptr, *ids = nullptr;
    unsigned long long *keysIn = nullptr, *keysOut = nullptr;
    void* temporary = nullptr;
    size_t temporaryBytes = 0;

    LeaderIndex(int count, bool active, bool useRadix)
        : n(count), enabled(active), radix(useRadix) {
        if (!enabled) return;
        if (n < 1) throw std::runtime_error("Leader index requires positive count");
        check(cudaMalloc(&order, n * sizeof(int)));
        check(cudaMalloc(&leaders, n * sizeof(int)));
        if (radix) {
            check(cudaMalloc(&ids, n * sizeof(int)));
            check(cudaMalloc(&keysIn, n * sizeof(*keysIn)));
            check(cudaMalloc(&keysOut, n * sizeof(*keysOut)));
            check(cub::DeviceRadixSort::SortPairs(nullptr, temporaryBytes,
                                                keysIn, keysOut, ids, order, n));
            check(cudaMalloc(&temporary, temporaryBytes));
        }
    }
    void prepare(const Car* cars) {
        if (radix) makeLeaderKeys<<<(n + 127) / 128, 128>>>(cars, keysIn, ids, n);
        else initializeLaneOrder<<<(n + 127) / 128, 128>>>(order, n);
        check(cudaGetLastError());
    }
    void sort(const Car* cars) {
        if (radix) check(cub::DeviceRadixSort::SortPairs(temporary, temporaryBytes,
                                                       keysIn, keysOut, ids, order, n));
        else thrust::sort(thrust::device_pointer_cast(order),
                          thrust::device_pointer_cast(order) + n, LanePositionLess{cars});
    }
    void link(const Car* cars) {
        linkLaneLeaders<<<(n + 127) / 128, 128>>>(cars, order, leaders, n);
        check(cudaGetLastError());
    }
    void build(const Car* cars) {
        if (enabled) { prepare(cars); sort(cars); link(cars); }
    }
    LeaderIndex(const LeaderIndex&) = delete;
    LeaderIndex& operator=(const LeaderIndex&) = delete;
    ~LeaderIndex() {
        if (order) cudaFree(order);
        if (leaders) cudaFree(leaders);
        if (ids) cudaFree(ids);
        if (keysIn) cudaFree(keysIn);
        if (keysOut) cudaFree(keysOut);
        if (temporary) cudaFree(temporary);
    }
};
