#include "hashing.h"

#include <iostream>  // For Debugging ONLY
#include <random>
#include <vector>

namespace hashing {

uint64_t random_bits(int8_t bits) {
  static std::random_device *rd = new std::random_device();
  static auto *gen = new std::mt19937_64((*rd)());
  std::uniform_int_distribution<uint64_t> dist(0, (UINT64_C(1) << bits) - 1);
  return dist(*gen);
}

uint64_t random_odd(int8_t bits) {
  if (bits == 0) {
    // There's no odd number with 0 bits, so we return 1.
    return 1;
  }
  uint64_t r = random_bits(bits - 1);
  return (r << 1) | 1;
}

bool checkCycleLength(uint64_t m, int minAllowedLength = 1<<20) {
  uint64_t a = UINT64_C(1);
  int i;
  for (i = 0; i < minAllowedLength; ++i) {
    a *= m;
    if (a == UINT64_C(1)) return false;
  }
  return true;
}

// Checks a simple bit-distribution condition: no byte can be all 0s or all 1s.
bool checkBits(uint64_t a) {
  uint8_t* start = reinterpret_cast<uint8_t*>(&a);
  for (size_t i = 0; i < sizeof(uint64_t); ++i) {
    uint8_t currentByte = start[i];
    if (currentByte == 0 || currentByte == 0xff) {
      return false;
    }
  }
  return true;
}

// Note: checkCycle is expensive, so it's disabled by default.
uint64_t niceRandomOdd(bool checkCycle = false, int8_t bits = 64) {
  uint64_t m;
  do {
    m = random_odd(bits);
  } while (!checkBits(m) || (checkCycle && !checkCycleLength(m)));
  return m;
}

// The id is used to generate different hash functions with the same code
// (by generating different multipliers and bias).
template <unsigned int id>
uint32_t hash64Tuple(const span<const uint32_t> tuple) {
  static const std::vector<uint64_t> *multipliers = []() {
    std::vector<uint64_t> *v = new std::vector<uint64_t>(64);
    for (int8_t i = 0; i < 64; ++i) {
      (*v)[i] = niceRandomOdd();
    }
    return v;
  }();
  static const uint64_t bias = niceRandomOdd();
  uint64_t result = bias;
  auto tupleIt = tuple.begin();
  auto multiplierIt = multipliers->begin();
  for (; tupleIt != tuple.end() && multiplierIt != multipliers->end();
       ++tupleIt, ++multiplierIt) {
    result += (*tupleIt) * (*multiplierIt);
  }
  return static_cast<uint32_t>(result >> 32);
}

uint64_t hash32Tuple(const span<const uint64_t> tuple) {
  span<const uint32_t> tuple32 = {
      reinterpret_cast<const uint32_t*>(tuple.data()),
      tuple.size() * sizeof(uint64_t) / sizeof(uint32_t)
  };
  return (static_cast<uint64_t>(hash64Tuple<0>(tuple32)) << 32) |
         hash64Tuple<1>(tuple32);
}

uint32_t hashSpan(span<const uint32_t> s, int8_t bits) {
  static constexpr uint64_t p = UINT64_C(1) << 61 - 1;
  static const uint64_t coefficient = niceRandomOdd(true, 61);
  uint64_t result = 0;
  
}


uint64_t hashSpan(span<const uint64_t> s, int8_t bits) {
  auto condensedSize = (s.size() + 31) >> 5;  // Divide by 32, rounding up.
  std::vector<uint64_t> condensed{condensedSize};
  for (int i = 0; i < condensedSize; ++i) {
    condensed[i] = hash64Tuple(s.subspan(i << 6));
  }
  static const uint64_t coefficient = niceRandomOdd(true);
  uint64_t result = 0;
  for (uint64_t a : condensed) {
    result = result * coefficient + a;
  }
  return result >> (64 - bits);
}


uint64_t hashInt(uint64_t h, int8_t bits) {
  static const std::vector<uint64_t> *multipliers = []() {
    std::vector<uint64_t> *v = new std::vector<uint64_t>(64);
    for (int8_t i = 0; i < 64; ++i) {
      (*v)[i] = niceRandomOdd();
    }
    return v;
  }();
  // std::cout << "h: " << h << " bits: " << (int)bits << std::endl;
  uint64_t a = (*multipliers)[bits];
  // std::cout << "a: " << a << std::endl;
  // std::cout << "h * a: " << (h * a) << std::endl;
  // std::cout << "64 - bits: " << (64 - bits) << std::endl;
  uint64_t result = (h * a) >> (64 - bits);
  // std::cout << "result: " << result << std::endl;
  return result;
}

}  // namespace hashing