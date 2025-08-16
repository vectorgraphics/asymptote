// Minimal usage example: prints a hash. Tested on x86, ppc, arm.

#include <algorithm>
#include <cstring>
#include <iostream>

#include "highwayhash/highwayhash.h"

using namespace highwayhash;

int main(int argc, char* argv[]) {
  // We read from the args on purpose, to ensure a compile time constant will
  // not be used, for verifying assembly on the supported platforms.
  if (argc != 2) {
    std::cout << "Please provide 1 argument with a text to hash" << std::endl;
    return 1;
  }

  // Please use a different key to ensure your hashes aren't identical.
  HH_ALIGNAS(32) const HHKey key = {1, 2, 3, 4};

  // Aligning inputs to 32 bytes may help but is not required.
  const char* in = argv[1];
  const size_t size = strlen(in);

  // Type determines the hash size; can also be HHResult128 or HHResult256.
  HHResult64 result;

  // HH_TARGET_PREFERRED expands to the best specialization available for the
  // CPU detected via compiler flags (e.g. AVX2 #ifdef __AVX2__).
  HHStateT<HH_TARGET_PREFERRED> state(key);
  HighwayHashT(&state, in, size, &result);
  std::cout << "Hash   : " << result << std::endl;

  HighwayHashCatT<HH_TARGET_PREFERRED> cat(key);
  cat.Append(in, size);
  cat.Finalize(&result);
  std::cout << "HashCat: " << result << std::endl;
  return 0;
}
