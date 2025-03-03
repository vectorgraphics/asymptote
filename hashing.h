#include <cstdint>
#include <array>

#include "common.h"

namespace hashing {

uint64_t hashSpan(span<const uint64_t> s);
uint64_t hashSpan(span<const char> s);
uint64_t hashInt(uint64_t i);
std::array<uint64_t, 4> fingerprint(span<const char> s);

}  // namespace hashing