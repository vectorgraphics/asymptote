
#ifndef RANDOM_H
#define RANDOM_H

#include <cstdint>
#include <random>
#include <string>

namespace camp {
void seed(int64_t seed);
int64_t randInt(int64_t min, int64_t max);
double unitrand();
}  // namespace camp

#endif
