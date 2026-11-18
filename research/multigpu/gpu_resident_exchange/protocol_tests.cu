#define LPSIM_RESIDENT_LIBRARY
#include "resident.cu"

using namespace resident;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

struct Fixture {
    Slots slots;
    Car* cars;
    Message* messages;
    Envelope* heads;
    Fixture() {
        std::vector<Car> empty(2); for (auto& c : empty) c.state = -1;
        cars = device(empty);
        slots = {2, 8, 0, 2, device(std::vector<int>(2, -1)), device(std::vector<int>(8, -1)),
            device(std::vector<int>(2, 0)), device(std::vector<int>{0, 1}), device(std::vector<int>{2}),
            device(std::vector<int>{0}), device(std::vector<unsigned long long>(2))};
        check(cudaMalloc(&messages, 4 * sizeof(Message)));
        check(cudaMalloc(&heads, sizeof(Envelope)));
    }
    ~Fixture() {
        for (void* p : {static_cast<void*>(cars), static_cast<void*>(messages), static_cast<void*>(heads),
             static_cast<void*>(slots.ids), static_cast<void*>(slots.lookup), static_cast<void*>(slots.owners),
             static_cast<void*>(slots.freeSlots), static_cast<void*>(slots.freeCount), static_cast<void*>(slots.error),
             static_cast<void*>(slots.refreshed)}) cudaFree(p);
    }
    int value(int* p) { int result; check(cudaMemcpy(&result, p, 4, cudaMemcpyDeviceToHost)); return result; }
    void receive(std::vector<Message> batch, unsigned long long epoch, int phase = 0,
                 unsigned long long expected = 0) {
        Envelope h{epoch, phase, 1, 0, int(batch.size()), 0};
        check(cudaMemcpy(heads, &h, sizeof(h), cudaMemcpyHostToDevice));
        if (!batch.empty()) check(cudaMemcpy(messages, batch.data(), batch.size() * sizeof(Message), cudaMemcpyHostToDevice));
        pullState<<<1, 32>>>(heads, messages, 4, cars, slots, 1, expected ? expected : epoch, phase);
        check(cudaDeviceSynchronize());
    }
};
Message message(int id, float position = 0) { Car c{}; c.state = 1; c.n = 2; c.pos = position; return {id, 1, c}; }

int main() { try {
    check(cudaSetDevice(0));
    { Fixture f; f.receive({}, 1); require(f.value(f.slots.error) == 0, "Empty publication failed"); }
    { Fixture f; f.receive({message(3)}, 1); int slot = f.value(f.slots.lookup + 3);
      require(slot >= 0 && f.value(f.slots.freeCount) == 1, "Insertion failed");
      f.receive({message(3, 10)}, 2, 1);
      require(f.value(f.slots.lookup + 3) == slot && f.value(f.slots.freeCount) == 1, "Refresh changed membership");
      Car car; check(cudaMemcpy(&car, f.cars + slot, sizeof(Car), cudaMemcpyDeviceToHost));
      require(car.pos == 10, "Refresh retained stale payload");
      retireSlots<<<1, 32>>>(f.cars, f.slots, 2); check(cudaDeviceSynchronize());
      require(f.value(f.slots.lookup + 3) == slot, "Fresh ghost removed");
      retireSlots<<<1, 32>>>(f.cars, f.slots, 3); check(cudaDeviceSynchronize());
      require(f.value(f.slots.lookup + 3) == -1 && f.value(f.slots.freeCount) == 2, "Obsolete ghost retained");
      f.receive({message(4)}, 4); require(f.value(f.slots.error) == 0, "Slot reuse failed"); }
    { Fixture f; f.receive({message(3), message(3)}, 1);
      require(f.value(f.slots.error) == Duplicate, "Duplicate accepted"); }
    { Fixture f; f.receive({message(1), message(2), message(3)}, 1);
      require(f.value(f.slots.error) == ReceiveOverflow, "Overflow accepted"); }
    { Fixture f; f.receive({message(1)}, 1, 0, 3);
      require(f.value(f.slots.error) == BadEnvelope, "Stale epoch accepted"); }
    { Fixture f; auto m = message(1); m.owner = 0; f.receive({m}, 2, 1);
      require(f.value(f.slots.error) == BadOwner, "Wrong-phase migration accepted"); }
    std::cout << "PASS protocol: empty, insert, refresh, retire, reuse, duplicate, overflow, epoch, ownership\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
