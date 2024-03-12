#include "hashing.h"

#include <iostream>  // For Debugging ONLY
#include <algorithm>
#include <random>
#include <vector>

#include "highwayhash/highwayhash_target.h"
#include "highwayhash/instruction_sets.h"

namespace hashing {
using namespace highwayhash;

// uint64_t highwayHash() {
//   HH_ALIGNAS(32) const HHKey key = {1, 2, 3, 4};
//   char in[8] = {1};
//   HHResult64 result;  // or HHResult128 or HHResult256
//   InstructionSets::Run<HighwayHash>(key, in, 8, &result);
//   return result;
// }


uint64_t random_bits(int8_t bits) {
  static std::random_device *rd = new std::random_device();
  static auto *gen = new std::mt19937_64((*rd)());
  std::uniform_int_distribution<uint64_t> dist(0, (UINT64_C(1) << bits) - 1);
  return dist(*gen);
}

uint64_t hashSpan(span<const char> s, int8_t bits) {
  HH_ALIGNAS(32) static const HHKey key = {random_bits(64), random_bits(64),
                                           random_bits(64), random_bits(64)};
  HHResult64 result;
  InstructionSets::Run<HighwayHash>(key, s.data(), s.size(), &result);
  return result & ((UINT64_C(1) << bits) - 1);
}

uint64_t hashSpan(span<const uint64_t> s, int8_t bits) {
  span<const char> sChar = {reinterpret_cast<const char*>(s.data()),
                            s.size() * (sizeof(uint64_t) / sizeof(char))};
  return hashSpan(sChar, bits);
}

std::array<uint64_t, 4> fingerprint(span<const char> s) {
  // The following key was generated using the Python `secrets` module.
  // However, since the key is public, the resulting hash is not secure.
  // (While HighwayHash makes cryptographic claims, those claims rely on
  // the secrecy of the key.)
  HH_ALIGNAS(32) static constexpr HHKey key= {
      UINT64_C(0x6e1b31ab5e83c15a),
      UINT64_C(0x6648d2208b67c4af),
      UINT64_C(0xcddc6e8f557f7103),
      UINT64_C(0x0729a6dd6e86d99a)
  };
  HHResult256 result;
  InstructionSets::Run<HighwayHash>(key, s.data(), s.size(), &result);
  std::array<uint64_t, 4> fingerprint;
  std::copy_n(result, 4, fingerprint.begin());
  return fingerprint;
}

uint64_t hashInt(uint64_t i, int8_t bits) {
  static const uint64_t key = random_bits(64);
  uint64_t mask = (UINT64_C(1) << bits) - 1;
  uint64_t currentKey = key & mask;
  return (i >> (64 - bits)) ^ (i & mask) ^ currentKey;
}


}  // namespace hashing