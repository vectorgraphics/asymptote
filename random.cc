#include "random.h"

namespace {
uint64_t makeRandomSeed() {
  std::random_device rd;
  std::uniform_int_distribution<uint64_t> dist;
  return dist(rd);
}

std::mt19937_64 randEngine(makeRandomSeed());
}

namespace camp {

void seed(int64_t seed) {
  uint64_t unsignedSeed;
  if (seed < 0) {
    unsignedSeed = makeRandomSeed();
  } else {
    unsignedSeed = static_cast<uint64_t>(seed);
  }
  randEngine=std::mt19937_64(unsignedSeed);
}

int64_t randInt(int64_t min, int64_t max) {
  std::uniform_int_distribution<int64_t> dist(min, max);
  return dist(randEngine);
}

double unitrand() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(randEngine);
}

}  // namespace camp
