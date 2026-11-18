#pragma once
// Included after the physical-junction Car/Road/Movement types and indication().
#include <climits>
#include <map>
#include <tuple>
#include <thrust/system/cuda/execution_policy.h>

namespace mobil {
struct Config {
    bool enabled = false;
    float politeness = .3f, safeBraking = 4.f, threshold = .2f, junctionMargin = 20.f;
    static Config environment() {
        Config c;
        if (const char* v = std::getenv("LPSIM_MOBIL")) {
            if (std::string(v) != "0" && std::string(v) != "1") throw std::runtime_error("LPSIM_MOBIL must be 0 or 1");
            c.enabled = std::string(v) == "1";
        }
        auto read = [](const char* name, float& value) {
            if (const char* s = std::getenv(name)) {
                std::string text(s); size_t used; value = std::stof(text, &used);
                if (used != text.size() || !std::isfinite(value)) throw std::runtime_error(std::string("Invalid ") + name);
            }
        };
        read("LPSIM_MOBIL_POLITENESS", c.politeness); read("LPSIM_MOBIL_SAFE_BRAKING", c.safeBraking);
        read("LPSIM_MOBIL_THRESHOLD", c.threshold); read("LPSIM_MOBIL_JUNCTION_MARGIN", c.junctionMargin);
        if (c.politeness < 0 || c.politeness > 1 || c.safeBraking <= 0 || c.threshold <= 0 || c.junctionMargin < SPACING)
            throw std::runtime_error("Invalid MOBIL parameters");
        return c;
    }
};
struct Neighbors { int front = -1, back = -1; };
struct Proposal { int lane = -1, movement = -1; float gain = 0; };
__host__ __device__ inline int idAt(const int* ids, int i) { return ids ? ids[i] : i; }
// Same IDM parameters as physical-junction motion. Net gap excludes vehicle length.
__host__ __device__ inline float idm(float v, float leaderV, float gap, float desiredSpeed) {
    float desired = 1.38075f + fmaxf(0.f, v * .5433028f + v * (v - leaderV) / (2 * sqrtf(.5570409f * 2.902058f)));
    return .5570409f * (1 - powf(v / desiredSpeed, 4) - powf(desired / fmaxf(.05f, gap), 2));
}
__host__ __device__ inline float acceleration(const Car& c, int leader, const Car* cars,
        int movement, const Road* roads, const Movement* moves, float t, int mode) {
    const Road road = roads[c.edges[c.k]];
    bool open = c.k == c.n - 1 || indication(moves[movement], t, mode) == 2;
    float gap = open ? 1.e6f : road.length - c.pos + 2.f, v = 0;
    if (leader >= 0 && cars[leader].pos - c.pos - 4.5f < gap) {
        gap = cars[leader].pos - c.pos - 4.5f; v = cars[leader].v;
    }
    return idm(c.v, v, gap, road.speed);
}
__host__ __device__ inline int currentMovement(const Car& c) { return c.k + 1 < c.n ? c.moves[c.k] : -1; }
__host__ __device__ inline bool eligible(const Car& c, const Road* roads, Config cfg) {
    return c.state == 1 && roads[c.edges[c.k]].lanes > 1 && c.pos >= cfg.junctionMargin &&
        roads[c.edges[c.k]].length - c.pos >= cfg.junctionMargin;
}
__host__ __device__ inline Proposal evaluate(const Car* cars, int i, int target, int movement,
        Neighbors old, Neighbors next, const Road* roads, const Movement* moves, Config cfg, float t, int mode) {
    const Car& c = cars[i];
    if ((next.front >= 0 && cars[next.front].pos - c.pos < SPACING) ||
        (next.back >= 0 && c.pos - cars[next.back].pos < SPACING)) return {};
    float before = acceleration(c, old.front, cars, currentMovement(c), roads, moves, t, mode);
    float after = acceleration(c, next.front, cars, movement, roads, moves, t, mode);
    float others = 0;
    if (next.back >= 0) {
        const Car& follower = cars[next.back];
        float afterBack = acceleration(follower, i, cars, currentMovement(follower), roads, moves, t, mode);
        if (afterBack < -cfg.safeBraking) return {};
        others += afterBack - acceleration(follower, next.front, cars, currentMovement(follower), roads, moves, t, mode);
    }
    if (old.back >= 0) {
        const Car& follower = cars[old.back];
        others += acceleration(follower, old.front, cars, currentMovement(follower), roads, moves, t, mode) -
                  acceleration(follower, i, cars, currentMovement(follower), roads, moves, t, mode);
    }
    float gain = after - before + cfg.politeness * others;
    return gain > cfg.threshold ? Proposal{target, movement, gain} : Proposal{};
}

// A candidate may only select a connector preserving its next road AND next lane.
// This leaves future route segments valid; roads without that connector reject it.
inline std::vector<int> legalMovements(const std::vector<Movement>& moves) {
    std::map<std::tuple<int,int,int,int>, int> first;
    for (int m = 0; m < int(moves.size()); ++m)
        first.emplace(std::make_tuple(moves[m].in, moves[m].out, moves[m].outLane, moves[m].inLane), m);
    std::vector<int> legal(moves.size() * 8, -1);
    for (int m = 0; m < int(moves.size()); ++m) for (int lane = 0; lane < 8; ++lane) {
        auto it = first.find(std::make_tuple(moves[m].in, moves[m].out, moves[m].outLane, lane));
        if (it != first.end()) legal[m * 8 + lane] = it->second;
    }
    return legal;
}
struct LaneLess {
    const Car* cars; const int* ids;
    __device__ bool operator()(int a, int b) const {
        const Car &x = cars[a], &y = cars[b];
        int kx = x.state == 1 ? x.edges[x.k] * 8 + x.lanes[x.k] : INT_MAX;
        int ky = y.state == 1 ? y.edges[y.k] * 8 + y.lanes[y.k] : INT_MAX;
        if (kx != ky) return kx < ky;
        if (kx != INT_MAX && x.pos != y.pos) return x.pos < y.pos;
        return idAt(ids, a) != idAt(ids, b) ? idAt(ids, a) < idAt(ids, b) : a < b;
    }
};
__global__ void initialize(int* order, int n, int* begin, int* end, int* winners, int nr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) order[i] = i;
    if (i < nr * 8) begin[i] = end[i] = 0;
    if (i < nr) winners[i] = INT_MAX;
}
__global__ void ranges(const Car* cars, const int* order, int n, int* begin, int* end) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return; const Car& c = cars[order[k]]; if (c.state != 1) return;
    int key = c.edges[c.k] * 8 + c.lanes[c.k];
    if (k == 0 || cars[order[k-1]].state != 1 || cars[order[k-1]].edges[cars[order[k-1]].k] * 8 + cars[order[k-1]].lanes[cars[order[k-1]].k] != key) begin[key] = k;
    if (k + 1 == n || cars[order[k+1]].state != 1 || cars[order[k+1]].edges[cars[order[k+1]].k] * 8 + cars[order[k+1]].lanes[cars[order[k+1]].k] != key) end[key] = k + 1;
}
__device__ inline Neighbors indexed(const Car* cars, int i, int lane, const int* order, const int* begin, const int* end) {
    const Car& c = cars[i]; int key = c.edges[c.k] * 8 + lane, lo = begin[key], hi = end[key];
    while (lo < hi) { int mid = (lo + hi) / 2; if (cars[order[mid]].pos < c.pos) lo = mid + 1; else hi = mid; }
    int front = lo; if (front < end[key] && order[front] == i) ++front;
    return {front < end[key] ? order[front] : -1, lo > begin[key] ? order[lo-1] : -1};
}
__global__ void propose(const Car* cars, int n, const Road* roads, const Movement* moves,
        const int* legal, const int* ids, const int* owners, int rank, Config cfg, float t, int mode,
        const int* order, const int* begin, const int* end, Proposal* proposals, int* winners) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    proposals[i] = {};
    if ((owners && owners[i] != rank) || !eligible(cars[i], roads, cfg)) return;
    const Car& c = cars[i]; int lane = c.lanes[c.k]; Proposal best;
    Neighbors old = indexed(cars, i, lane, order, begin, end);
    for (int target : {lane - 1, lane + 1}) {
        if (target < 0 || target >= roads[c.edges[c.k]].lanes) continue;
        int movement = c.k + 1 < c.n ? legal[c.moves[c.k] * 8 + target] : -1;
        if (c.k + 1 < c.n && movement < 0) continue;
        Proposal p = evaluate(cars, i, target, movement, old, indexed(cars, i, target, order, begin, end), roads, moves, cfg, t, mode);
        if (p.lane >= 0 && (best.lane < 0 || p.gain > best.gain)) best = p;
    }
    proposals[i] = best;
    if (best.lane >= 0) atomicMin(winners + c.edges[c.k], idAt(ids, i));
}
__global__ void commit(Car* cars, int n, const int* ids, const Proposal* proposals, const int* winners, unsigned long long* changes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const Proposal p = proposals[i]; if (p.lane < 0) return;
    Car& c = cars[i]; if (winners[c.edges[c.k]] != idAt(ids, i)) return;
    c.lanes[c.k] = p.lane; if (c.k + 1 < c.n) c.moves[c.k] = p.movement;
    atomicAdd(changes, 1ULL);
}

class Engine {
    int n, nr; int *order = nullptr, *begin = nullptr, *end = nullptr, *winner = nullptr, *legal = nullptr;
    Proposal* proposals = nullptr; unsigned long long* changes = nullptr;
public:
    Config config;
    Engine(int count, int roads, const std::vector<Movement>& moves, Config cfg = Config::environment()) : n(count), nr(roads), config(cfg) {
        if (!config.enabled) return;
        order = device(std::vector<int>(n)); begin = device(std::vector<int>(nr*8)); end = device(std::vector<int>(nr*8));
        winner = device(std::vector<int>(nr)); legal = device(legalMovements(moves));
        proposals = device(std::vector<Proposal>(n)); changes = device(std::vector<unsigned long long>{0});
    }
    Engine(const Engine&) = delete; Engine& operator=(const Engine&) = delete;
    ~Engine() { for (int* p : {order, begin, end, winner, legal}) if (p) cudaFree(p); if (proposals) cudaFree(proposals); if (changes) cudaFree(changes); }
    void step(Car* cars, const Road* roads, const Movement* moves, float t, int mode,
              const int* ids = nullptr, const int* owners = nullptr, int rank = -1) {
        if (!config.enabled) return;
        initialize<<<(std::max(n,nr*8)+127)/128,128>>>(order,n,begin,end,winner,nr);
        thrust::sort(thrust::cuda::par_nosync.on(0),order,order+n,LaneLess{cars,ids});
        ranges<<<(n+127)/128,128>>>(cars,order,n,begin,end);
        propose<<<(n+127)/128,128>>>(cars,n,roads,moves,legal,ids,owners,rank,config,t,mode,order,begin,end,proposals,winner);
        commit<<<(n+127)/128,128>>>(cars,n,ids,proposals,winner,changes); check(cudaGetLastError());
    }
    unsigned long long count() const { unsigned long long out = 0; if (changes) check(cudaMemcpy(&out,changes,sizeof(out),cudaMemcpyDeviceToHost)); return out; }
};

// Serial verification reference: brute-force neighbors, serial stable-ID selection.
// Shares model equations, not GPU indexing, arbitration atomics or storage layout.
inline unsigned long long reference(std::vector<Car>& cars, const std::vector<Road>& roads,
        const std::vector<Movement>& moves, Config cfg, float t, int mode) {
    if (!cfg.enabled) return 0;
    auto legal = legalMovements(moves); std::vector<Proposal> proposals(cars.size());
    std::vector<int> winner(roads.size(), INT_MAX);
    auto neighbors = [&](int i, int lane) {
        Neighbors out; const Car& c = cars[i];
        for (int j = 0; j < int(cars.size()); ++j) {
            const Car& d = cars[j]; if (i == j || d.state != 1 || d.edges[d.k] != c.edges[c.k] || d.lanes[d.k] != lane) continue;
            if (d.pos >= c.pos && (out.front < 0 || d.pos < cars[out.front].pos || (d.pos == cars[out.front].pos && j < out.front))) out.front = j;
            if (d.pos < c.pos && (out.back < 0 || d.pos > cars[out.back].pos || (d.pos == cars[out.back].pos && j > out.back))) out.back = j;
        }
        return out;
    };
    for (int i = 0; i < int(cars.size()); ++i) {
        const Car& c = cars[i]; if (!eligible(c,roads.data(),cfg)) continue;
        for (int lane : {c.lanes[c.k]-1,c.lanes[c.k]+1}) {
            if (lane < 0 || lane >= roads[c.edges[c.k]].lanes) continue;
            int movement = c.k+1<c.n ? legal[c.moves[c.k]*8+lane] : -1;
            if (c.k+1<c.n && movement<0) continue;
            auto p = evaluate(cars.data(),i,lane,movement,neighbors(i,c.lanes[c.k]),neighbors(i,lane),roads.data(),moves.data(),cfg,t,mode);
            if (p.lane>=0 && (proposals[i].lane<0 || p.gain>proposals[i].gain)) proposals[i]=p;
        }
        if (proposals[i].lane>=0) winner[c.edges[c.k]]=std::min(winner[c.edges[c.k]],i);
    }
    unsigned long long count=0;
    for (int i=0;i<int(cars.size());++i) if (proposals[i].lane>=0 && winner[cars[i].edges[cars[i].k]]==i) {
        Car& c=cars[i]; c.lanes[c.k]=proposals[i].lane; if(c.k+1<c.n)c.moves[c.k]=proposals[i].movement; ++count;
    }
    return count;
}
}
