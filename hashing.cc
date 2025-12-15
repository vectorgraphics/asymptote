#include "hashing.h"

#include <iostream>  // For Debugging ONLY
#include <algorithm>
#include <random>
#include <vector>

#include <highwayhash/highwayhash_target.h>
#include <highwayhash/instruction_sets.h>

namespace hashing {
using namespace highwayhash;

// uint64_t highwayHash() {
//   HH_ALIGNAS(32) const HHKey key = {1, 2, 3, 4};
//   char in[8] = {1};
//   HHResult64 result;  // or HHResult128 or HHResult256
//   InstructionSets::Run<HighwayHash>(key, in, 8, &result);
//   return result;
// }

uint64_t constexpr shiftLeftDefined(uint64_t x, int8_t shift) {
  return shift >= 64 ? 0 : x << shift;
}

uint64_t random_bits(int8_t bits) {
  static std::random_device *rd = new std::random_device();
  static auto *gen = new std::mt19937_64((*rd)());
  // uint64_t max = (bits >= 64 ? UINT64_C(-1) : (UINT64_C(1) << bits) - 1);
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
