#include "hashing.h"

#include <algorithm>
#include <random>
#include <vector>

#include <highwayhash/highwayhash_target.h>
#include <highwayhash/instruction_sets.h>

namespace hashing {
using namespace highwayhash;

uint64_t constexpr shiftLeftDefined(uint64_t x, uint8_t shift) {
  return shift >= 64 ? 0 : x << shift;
}

uint64_t random_bits(uint8_t bits) {
  static std::random_device *rd = new std::random_device();
  static auto *gen = new std::mt19937_64((*rd)());
  std::uniform_int_distribution<uint64_t> dist(
    0, shiftLeftDefined(1, bits) - 1);
  return dist(*gen);
}

uint64_t hashSpan(span<const char> s) {
  HH_ALIGNAS(32) static const HHKey key = {random_bits(64), random_bits(64),
                                           random_bits(64), random_bits(64)};
  HHResult64 result;
  InstructionSets::Run<HighwayHash>(key, s.data(), s.size(), &result);
  return result & (shiftLeftDefined(1, 62) - 1);
}

uint64_t hashSpan(span<const uint64_t> s) {
  span<const char> sChar = {reinterpret_cast<const char*>(s.data()),
                            s.size() * (sizeof(uint64_t) / sizeof(char))};
  return hashSpan(sChar);
}

uint64_t hashInt(uint64_t i) {
  span<const uint64_t> s = {&i, 1};
  return hashSpan(s);
}


}  // namespace hashing
