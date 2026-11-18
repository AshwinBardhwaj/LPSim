// Experimental persistent-slot backend. Frozen reference is included, not edited.
#define main frozen_correctness_main
#include "../source/research/multigpu/partitioned.cu"
#undef main
#include <thrust/system/cuda/execution_policy.h>
#include <filesystem>
#include "physics_adapters.cuh"
#include "slots.cuh"
#include "mobil.cuh"

namespace resident {
struct Rank {
    std::unique_ptr<Partition> p;
    Slots slots;
    mobil::Config mobilConfig = mobil::Config::environment();
    std::unique_ptr<mobil::Engine> mobilEngine;
    Message* messages[3];
    Envelope* heads[3];
    cudaEvent_t published[3], consumed[3], reservationReady, reservationConsumed;
    ReservationMessage* reservations;
    int* reservationKeys;
    std::vector<int> offsets, lengths;
    std::vector<int> neighbors;
    int reservationCount = 0;
    int *laneOrder, *leaders;
    int *interiorLanes, *boundaryLanes;
    int interiorCount = 0, boundaryCount = 0;
    bool rebuild = std::getenv("LPSIM_SLOT_REBUILD") && std::string(std::getenv("LPSIM_SLOT_REBUILD")) == "1";
    Car* gathered = nullptr;
    int *gatherIDs = nullptr, *gatherOwners = nullptr;
    unsigned long long* gatherEpochs = nullptr;
    Rank(int rank, int gpu, int ranks, const Model& model, const std::vector<Car>& initial,
         const std::vector<int>& jo, int capacityOverride) {
        std::vector<Car> visible;
        std::vector<int> ids, owners;
        std::vector<int> lookup(initial.size(), -1);
        for (int id = 0; id < int(initial.size()); ++id) {
            const Car& c = initial[id]; int owner = authority(c, model.roads.data(), model.moves.data(), jo.data());
            if (owner == rank || (c.state == 1 && jo[model.roads[c.edges[c.k]].from] == rank)) {
                lookup[id] = ids.size(); ids.push_back(id); owners.push_back(owner); visible.push_back(c);
            }
        }
        int live = ids.size();
        int cap = capacityOverride ? capacityOverride : std::min(int(initial.size()), std::max(256, 2 * live));
        if (cap < live || cap < 1) throw std::runtime_error("Insufficient slot capacity");
        visible.resize(cap); ids.resize(cap, -1); owners.resize(cap, rank);
        for (int i = live; i < cap; ++i) visible[i].state = -1;
        p.reset(new Partition(rank, gpu, model, visible, owners, jo, false, 1));
        std::vector<int> freeSlots(cap, -1);
        for (int i = live; i < cap; ++i) freeSlots[i - live] = i;
        slots = {cap, int(initial.size()), rank, ranks, device(ids), device(lookup), p->owners,
                 device(freeSlots), device(std::vector<int>{cap - live}), device(std::vector<int>{0}),
                 device(std::vector<unsigned long long>(cap, 0))};
        laneOrder = device(std::vector<int>(cap)); leaders = device(std::vector<int>(cap));
        if (rebuild) {
            check(cudaMalloc(&gathered, cap * sizeof(Car)));
            gatherIDs = device(std::vector<int>(cap)); gatherOwners = device(std::vector<int>(cap));
            gatherEpochs = device(std::vector<unsigned long long>(cap));
        }
        mobilEngine.reset(new mobil::Engine(cap, model.nr, model.moves, mobilConfig));
        for (int phase = 0; phase < (mobilConfig.enabled ? 3 : 2); ++phase) {
            check(cudaMalloc(&messages[phase], size_t(ranks) * cap * sizeof(Message)));
            check(cudaMalloc(&heads[phase], ranks * sizeof(Envelope)));
            check(cudaEventCreateWithFlags(&published[phase], cudaEventDisableTiming));
            check(cudaEventCreateWithFlags(&consumed[phase], cudaEventDisableTiming));
        }
        check(cudaEventCreateWithFlags(&reservationReady, cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&reservationConsumed, cudaEventDisableTiming));
        offsets.resize(ranks); lengths.resize(ranks);
        std::vector<int> keys;
        for (int d = 0; d < ranks; ++d) {
            offsets[d] = keys.size();
            bool neighbor = false;
            for (int e = 0; e < model.nr; ++e) {
                int a = jo[model.roads[e].from], b = jo[model.roads[e].to];
                if (d != rank && ((a == rank && b == d) || (a == d && b == rank))) neighbor = true;
                if (d != rank && a == rank && b == d)
                    for (int lane = 0; lane < model.roads[e].lanes; ++lane) keys.push_back(e * 8 + lane);
            }
            lengths[d] = keys.size() - offsets[d];
            if (neighbor) neighbors.push_back(d);
        }
        reservationCount = keys.size();
        reservationKeys = device(keys.empty() ? std::vector<int>{0} : keys);
        check(cudaMalloc(&reservations, std::max(1, reservationCount) * sizeof(ReservationMessage)));
        std::vector<int> localLanes, remoteLanes;
        for (int road = 0; road < model.nr; ++road) if (jo[model.roads[road].to] == rank)
            for (int lane = 0; lane < model.roads[road].lanes; ++lane)
                (jo[model.roads[road].from] == rank ? localLanes : remoteLanes).push_back(road * 8 + lane);
        interiorCount = localLanes.size(); boundaryCount = remoteLanes.size();
        interiorLanes = device(localLanes.empty() ? std::vector<int>{0} : localLanes);
        boundaryLanes = device(remoteLanes.empty() ? std::vector<int>{0} : remoteLanes);
    }
    void rebuildView() {
        if (!rebuild) return;
        int n = slots.capacity;
        initializeLaneOrder<<<(n + 127) / 128, 128>>>(laneOrder, n);
        thrust::sort(thrust::cuda::par_nosync.on(0), laneOrder, laneOrder + n, StableSlotLess{slots.ids});
        gatherSlotView<<<(n + 127) / 128, 128>>>(p->b, gathered, slots, laneOrder, gatherIDs, gatherOwners, gatherEpochs);
        check(cudaMemsetAsync(slots.freeCount, 0, sizeof(int)));
        installSlotView<<<(n + 127) / 128, 128>>>(p->b, gathered, slots, gatherIDs, gatherOwners, gatherEpochs);
    }
    void motion(float t) {
        p->select(); int n = slots.capacity;
        check(cudaMemcpyAsync(p->b, p->a, n * sizeof(Car), cudaMemcpyDeviceToDevice));
        initializeLaneOrder<<<(n + 127) / 128, 128>>>(laneOrder, n);
        thrust::sort(thrust::cuda::par_nosync.on(0), laneOrder, laneOrder + n, SlotLaneLess{p->a, slots.ids});
        linkLaneLeaders<<<(n + 127) / 128, 128>>>(p->a, laneOrder, leaders, n);
        p->context->build(p->a, p->moves, p->csr, t, 0, false);
        extractCompact<<<(n + 127) / 128, 128>>>(p->a, p->hot);
        advanceSlots<<<(n + 127) / 128, 128>>>(slots.ids, p->hot, p->b, n, p->roads, p->moves,
            p->csr, p->nm, t, .25f, 0, false, leaders, slots.owners, slots.rank,
            p->context->head, p->context->links, p->context->available);
        prepareJunctions<<<(std::max(n, p->cap) + 127) / 128, 128>>>(p->b, n, p->moves, .25f, p->ds);
        check(cudaGetLastError());
    }
    void junction(float t) {
        p->select(); int n = slots.capacity;
        // Post-motion exchange changed visible cars; reset summaries without moving connectors again.
        prepareJunctions<<<(std::max(n, p->cap) + 127) / 128, 128>>>(p->b, n, p->moves, 0, p->ds);
        resetSpawnIDs<<<(p->cap + 127) / 128, 128>>>(p->ds);
        summarizeSlots<<<(n + 127) / 128, 128>>>(slots.ids, p->b, n, p->roads, p->moves, t, 0, p->ds);
        requestJunctionEntry<<<(n + 127) / 128, 128>>>(p->b, n, p->roads, p->moves, p->csr, p->nm, t, 0, false, p->ds, p->requests);
        filterRequests<<<(n + 127) / 128, 128>>>(p->requests, n, slots.owners, slots.rank, p->nm);
        auto* index = p->requestIndex.get();
        check(cudaMemsetAsync(index->begin, 0, p->nj * sizeof(int)));
        check(cudaMemsetAsync(index->end, 0, p->nj * sizeof(int)));
        makeSlotRequestKeys<<<(n + 127) / 128, 128>>>(p->requests, p->moves, slots.ids, index->keys, n);
        thrust::sort(thrust::cuda::par_nosync.on(0), index->keys, index->keys + n);
        findJunctionRequestRanges<<<(n + 127) / 128, 128>>>(index->keys, n, index->begin, index->end);
        if (p->junctionCount) arbitrateSlots<<<(p->junctionCount + 127) / 128, 128>>>(slots.lookup,
            p->b, n, p->roads, p->moves, p->csr, p->nm, p->nj, false, p->ds, p->requests,
            index->keys, index->begin, index->end, p->junctions, p->junctionCount);
        check(cudaGetLastError());
    }
};

// A stream waits only for neighbors that own data it consumes. The event also
// supplies publication ordering for peer loads; no remote atomic polling needed.
void stateExchange(std::vector<std::unique_ptr<Rank>>& ranks, int step, int phase) {
    auto epoch = sequence(step, Phase(phase));
    for (auto& r : ranks) {
        r->p->select();
        if (step) for (int d : r->neighbors) check(cudaStreamWaitEvent(0, ranks[d]->consumed[phase], 0));
        resetOutboxes<<<1, 32>>>(r->heads[phase], ranks.size(), r->slots.rank, epoch, phase, r->slots.error);
        packSlotsState<<<(r->slots.capacity + 127) / 128, 128>>>(r->p->b, r->slots, r->p->roads,
            r->p->moves, r->p->jo, r->messages[phase], r->heads[phase], phase);
        finalizeOutboxes<<<1, 32>>>(r->heads[phase], ranks.size(), r->slots.error);
        check(cudaEventRecord(r->published[phase]));
    }
    for (auto& r : ranks) {
        r->p->select();
        for (int s : r->neighbors) {
            auto& src = ranks[s];
            check(cudaStreamWaitEvent(0, src->published[phase], 0));
            pullState<<<(src->slots.capacity + 127) / 128, 128>>>(src->heads[phase], src->messages[phase],
                src->slots.capacity, r->p->b, r->slots, s, epoch, phase);
        }
        retireSlots<<<(r->slots.capacity + 127) / 128, 128>>>(r->p->b, r->slots, epoch);
        r->rebuildView();
        check(cudaEventRecord(r->consumed[phase]));
        check(cudaGetLastError());
    }
}

void reservationExchange(std::vector<std::unique_ptr<Rank>>& ranks, int step, bool overlap) {
    for (auto& r : ranks) {
        r->p->select();
        if (step) for (int d : r->neighbors) if (r->lengths[d])
            check(cudaStreamWaitEvent(0, ranks[d]->reservationConsumed, 0));
        if (r->reservationCount) packReservations<<<(r->reservationCount + 127) / 128, 128>>>(
            r->p->ds, r->reservationKeys, r->p->roads, r->p->jo, r->reservationCount,
            r->slots.rank, step, r->reservations);
        check(cudaEventRecord(r->reservationReady));
    }
    // Local receiving lanes have no remote reservation authority. Their spawn
    // writes are disjoint from boundary spawn and can precede peer dependencies.
    if (overlap) for (auto& r : ranks) {
        r->p->select();
        if (r->interiorCount) spawnSlots<<<(r->interiorCount + 127) / 128, 128>>>(r->p->b,
            r->slots, r->p->nr, r->p->ds, r->interiorLanes, r->interiorCount);
    }
    for (auto& r : ranks) {
        r->p->select();
        for (int s : r->neighbors) {
            auto& src = ranks[s]; int n = src->lengths[r->slots.rank];
            if (!n) continue;
            check(cudaStreamWaitEvent(0, src->reservationReady, 0));
            unpackReservations<<<(n + 127) / 128, 128>>>(r->p->ds,
                src->reservations + src->offsets[r->slots.rank], n);
        }
        check(cudaEventRecord(r->reservationConsumed));
        check(cudaGetLastError());
    }
}

void synchronize(std::vector<std::unique_ptr<Rank>>& ranks) {
    for (auto& r : ranks) { r->p->select(); check(cudaDeviceSynchronize()); }
}

std::vector<Car> inspect(std::vector<std::unique_ptr<Rank>>& ranks, const Model& model,
                         const std::vector<int>& jo, int total, Partition* oracle) {
    std::vector<Car> result(total); std::vector<int> seen(total);
    std::vector<std::vector<int>> ids(ranks.size());
    for (auto& r : ranks) {
        r->p->select(); int error;
        check(cudaMemcpy(&error, r->slots.error, sizeof(int), cudaMemcpyDeviceToHost));
        if (error) throw std::runtime_error("Device protocol error " + std::to_string(error));
        r->p->inspect(); auto& localIDs = ids[r->slots.rank]; localIDs.resize(r->slots.capacity);
        std::vector<int> owners(r->slots.capacity);
        check(cudaMemcpy(localIDs.data(), r->slots.ids, localIDs.size() * sizeof(int), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(owners.data(), r->slots.owners, owners.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int slot = 0; slot < r->slots.capacity; ++slot) {
            int id = localIDs[slot]; if (id < 0) continue;
            if (id >= total) throw std::runtime_error("Invalid stable ID");
            const Car& c = r->p->inspection[slot];
            if (owners[slot] != authority(c, model.roads.data(), model.moves.data(), jo.data()))
                throw std::runtime_error("Authority mismatch");
            if (owners[slot] == r->slots.rank) { ++seen[id]; result[id] = c; }
        }
    }
    if (oracle) oracle->inspect();
    for (int id = 0; id < total; ++id) {
        if (seen[id] != 1) throw std::runtime_error("Missing/duplicate authoritative vehicle");
        if (oracle && std::memcmp(&result[id], &oracle->inspection[id], sizeof(Car)))
            throw std::runtime_error("Reference mismatch ID " + std::to_string(id));
    }
    for (auto& r : ranks) {
        std::vector<int> present(total);
        for (int slot = 0; slot < r->slots.capacity; ++slot) {
            int id = ids[r->slots.rank][slot]; if (id < 0) continue;
            if (++present[id] != 1 || std::memcmp(&result[id], &r->p->inspection[slot], sizeof(Car)))
                throw std::runtime_error("Duplicate slot or stale ghost");
        }
        for (int id = 0; id < total; ++id) {
            const Car& c = result[id];
            bool visible = authority(c, model.roads.data(), model.moves.data(), jo.data()) == r->slots.rank ||
                (c.state == 1 && jo[model.roads[c.edges[c.k]].from] == r->slots.rank);
            if (present[id] != int(visible)) throw std::runtime_error("Missing/extra ghost membership");
        }
    }
    validatePhysical(result, model);
    return result;
}
}

#ifndef LPSIM_RESIDENT_LIBRARY
int main(int argc, char** argv) { try {
    if (argc != 10) throw std::runtime_error(
        "resident MODEL SNAPSHOT OWNERS STEPS OUT RANKS VERIFY barrier|neighbor CAPACITY(0=auto)");
    setenv("LPSIM_MOTION_CONTEXT", "3", 1);
    Model model(argv[1]); std::ifstream input(argv[2], std::ios::binary);
    int n = 0; float start = 0; input.read(reinterpret_cast<char*>(&n), 4); input.read(reinterpret_cast<char*>(&start), 4);
    if (!input || n < 1 || !std::isfinite(start)) throw std::runtime_error("Invalid snapshot header");
    std::vector<Car> initial(n); input.read(reinterpret_cast<char*>(initial.data()), n * sizeof(Car));
    if (!input) throw std::runtime_error("Truncated snapshot");
    int ranksCount = std::stoi(argv[6]), steps = std::stoi(argv[4]);
    bool verify = std::stoi(argv[7]) != 0; std::string schedule = argv[8]; int capacity = std::stoi(argv[9]);
    if (ranksCount < 1 || ranksCount > 3 || steps < 1 || capacity < 0 ||
        (schedule != "barrier" && schedule != "neighbor")) throw std::runtime_error("Invalid options");
    std::vector<int> jo(model.nj); std::ifstream map(argv[3]);
    for (int& owner : jo) if (!(map >> owner) || owner < 0 || owner >= ranksCount)
        throw std::runtime_error("Invalid ownership map");
    int extra; if (map >> extra) throw std::runtime_error("Excess ownership entries");
    for (const auto& c : initial) {
        if (c.n < 2 || c.n > MAX_ROUTE || c.k < 0 || c.k >= c.n || c.state < 0 || c.state > 3)
            throw std::runtime_error("Invalid vehicle route/state");
        for (int k = 0; k < c.n; ++k) if (c.edges[k] < 0 || c.edges[k] >= model.nr ||
            c.lanes[k] < 0 || c.lanes[k] >= model.roads[c.edges[k]].lanes)
            throw std::runtime_error("Invalid route edge/lane");
        for (int k = 0; k < c.n - 1; ++k) if (c.moves[k] < 0 || c.moves[k] >= model.nm)
            throw std::runtime_error("Invalid route movement");
        if (c.state == 2 && (c.mid < 0 || c.mid >= model.nm)) throw std::runtime_error("Invalid connector");
    }
    validatePhysical(initial, model);
    std::filesystem::path out(argv[5]);
    if (!std::filesystem::create_directory(out)) throw std::runtime_error("Output must be a new directory");
    bool logical = std::getenv("LPSIM_LOGICAL") != nullptr;
    for (int d = 0; d < ranksCount; ++d) {
        check(cudaSetDevice(logical ? 0 : d));
        if (!logical) for (int s = 0; s < ranksCount; ++s) if (s != d) {
            int access; check(cudaDeviceCanAccessPeer(&access, d, s));
            if (!access) throw std::runtime_error("CUDA peer access unavailable");
            auto e = cudaDeviceEnablePeerAccess(s, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else check(e);
        }
    }
    std::vector<std::unique_ptr<resident::Rank>> ranks;
    for (int r = 0; r < ranksCount; ++r) ranks.emplace_back(new resident::Rank(r, logical ? 0 : r,
        ranksCount, model, initial, jo, capacity));
    std::unique_ptr<Partition> oracle;
    if (verify) { std::vector<int> owners(n); for (int i = 0; i < n; ++i)
        owners[i] = authority(initial[i], model.roads.data(), model.moves.data(), jo.data());
        oracle.reset(new Partition(-1, 0, model, initial, owners, jo, false, 1)); }
    resident::synchronize(ranks);
    auto begin = std::chrono::steady_clock::now(); double verificationMs = 0;
    std::vector<Car> result;
    for (int step = 0; step < steps; ++step) {
        float t = start + step * .25f;
        for (auto& r : ranks) r->motion(t);
        if (schedule == "barrier") resident::synchronize(ranks);
        resident::stateExchange(ranks, step, 0);
        if (schedule == "barrier") resident::synchronize(ranks);
        for (auto& r : ranks) r->junction(t);
        if (schedule == "barrier") resident::synchronize(ranks);
        resident::reservationExchange(ranks, step, schedule == "neighbor");
        if (schedule == "barrier") resident::synchronize(ranks);
        for (auto& r : ranks) { r->p->select();
            if (schedule == "neighbor") {
                if (r->boundaryCount) resident::spawnSlots<<<(r->boundaryCount + 127) / 128, 128>>>(
                    r->p->b, r->slots, model.nr, r->p->ds, r->boundaryLanes, r->boundaryCount);
            } else resident::spawnSlots<<<(model.nr * 8 + 127) / 128, 128>>>(r->p->b, r->slots, model.nr, r->p->ds);
        }
        resident::stateExchange(ranks, step, 1);
        if (ranks[0]->mobilConfig.enabled) {
            for (auto& r : ranks) {
                r->p->select();
                r->mobilEngine->step(r->p->b,r->p->roads,r->p->moves,t+.25f,0,r->slots.ids,r->slots.owners,r->slots.rank);
            }
            // Updated current lane and connector are mutable full-Car state. Publish
            // them before any neighbor's next motion/context build can consume ghosts.
            resident::stateExchange(ranks, step, 2);
        }
        if (schedule == "barrier" || verify) resident::synchronize(ranks);
        if (verify) {
            auto v = std::chrono::steady_clock::now();
            oracle->motion(t, .25f, 0); oracle->junction(t, 0); oracle->spawn();
            if (ranks[0]->mobilConfig.enabled) {
                oracle->inspect();
                mobil::reference(oracle->inspection,model.roads,model.moves,ranks[0]->mobilConfig,t+.25f,0);
                check(cudaMemcpy(oracle->b,oracle->inspection.data(),n*sizeof(Car),cudaMemcpyHostToDevice));
            }
            result = resident::inspect(ranks, model, jo, n, oracle.get()); std::swap(oracle->a, oracle->b);
            verificationMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - v).count();
        }
        if (step + 1 < steps) for (auto& r : ranks) std::swap(r->p->a, r->p->b);
    }
    resident::synchronize(ranks);
    double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (!verify) result = resident::inspect(ranks, model, jo, n, nullptr);
    std::ofstream final(out / "final.bin", std::ios::binary); float end = start + steps * .25f;
    final.write(reinterpret_cast<char*>(&n), 4); final.write(reinterpret_cast<char*>(&end), 4);
    final.write(reinterpret_cast<const char*>(result.data()), n * sizeof(Car));
    std::ofstream metrics(out / "metrics.json");
    metrics << "{\"backend\":\"cuda_peer_pull_slots\",\"schedule\":\"" << schedule
        << "\",\"verified\":" << (verify ? "true" : "false") << ",\"logical\":" << (logical ? "true" : "false")
        << ",\"steps\":" << steps << ",\"ranks\":" << ranksCount << ",\"wall_ms\":" << wallMs
        << ",\"verification_ms\":" << verificationMs << ",\"count_d2h\":0,\"slot_rebuild\":"
        << (ranks[0]->rebuild ? "true" : "false") << ",\"slots\":[";
    for (int r = 0; r < ranksCount; ++r) metrics << (r ? "," : "") << ranks[r]->slots.capacity;
    unsigned long long laneChanges=0;
    for (auto& r : ranks) { r->p->select(); laneChanges+=r->mobilEngine->count(); }
    metrics << "],\"mobil_enabled\":" << (ranks[0]->mobilConfig.enabled ? "true" : "false")
            << ",\"mobil_changes\":" << laneChanges << "}\n";
    if (!final || !metrics) throw std::runtime_error("Output write failed");
    std::cout << "PASS resident " << steps << " steps; wall_ms=" << wallMs << " verified=" << verify << '\n';
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
#endif
