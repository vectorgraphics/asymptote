#include "hashing.h"

#include <algorithm>
#include <random>
#include <vector>

#include "wyhash/wyhash.h"

namespace hashing {

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

std::array<uint64_t, 4> make_secret_array(uint64_t seed) {
  std::array<uint64_t, 4> secret;
  make_secret(seed, secret.data());
  return secret;
}

uint64_t hashSpan(span<const char> s) {
  static const uint64_t seed = random_bits(64);
  static const std::array<uint64_t, 4> secret = make_secret_array(seed);
  uint64_t result = wyhash(s.data(), s.size(), seed, secret.data());
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
