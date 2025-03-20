
#include <random>
#include <string>

namespace camp_random {
void seed(uint64_t seed);
uint64_t randInt(uint64_t min, uint64_t max);
double unitrand();
}  // namespace camp_random