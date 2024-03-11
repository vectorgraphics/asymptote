#include <cstdint>

#include "common.h"

#if __cplusplus < 202002L
#include <boost/core/span.hpp>
using boost::span;
#else
#include <span>
using std::span;
#endif

namespace hashing {

uint64_t hashInt(uint64_t h, int8_t bits);
uint64_t hashSpan(span<const uint64_t> s, int8_t bits);

}  // namespace hashing