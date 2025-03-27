#include "random.h"

namespace {
std::minstd_rand randEngine(1);
}

namespace camp_random {

void seed(uint64_t seed) {
  randEngine=std::minstd_rand(seed);
}

uint64_t randInt(uint64_t min, uint64_t max) {
  std::uniform_int_distribution<uint64_t> dist(min, max);
  return dist(randEngine);
}
double unitrand() {
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  return dist(randEngine);
}

}  // namespace camp_random