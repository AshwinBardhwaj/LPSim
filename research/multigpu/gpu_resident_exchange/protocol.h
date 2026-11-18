#pragma once
#include <cstdint>
#include <limits>

namespace resident {
enum class Phase : int { Motion = 0, Spawn = 1, LaneChange = 2 };
enum Error : int { Ok = 0, SendOverflow = 1, ReceiveOverflow = 2,
                   BadEnvelope = 3, Duplicate = 4, WrongMigration = 5,
                   BadOwner = 6 };
struct Envelope {
    std::uint64_t epoch;
    int phase, source, destination, count, error;
};
// Epoch 0 is reserved for uninitialized storage. No wrapping sequence reuse.
constexpr std::uint64_t sequence(std::uint64_t step, Phase phase) {
    return 3 * step + static_cast<int>(phase) + 1;
}
}
