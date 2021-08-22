#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_THREAD 1

#ifdef _WIN32

#include "packages/zlib.v140.windesktop.msvcstl.dyn.rt-dyn.1.2.8.8/build/native/include/zlib.h"
#pragma comment(lib, "packages/zlib.v140.windesktop.msvcstl.dyn.rt-dyn.1.2.8.8/lib/native/v140/windesktop/msvcstl/dyn/rt-dyn/x64/Release/zlib.lib")

#else

#include "zlib.h"
#include <tinyexr/tinyexr.h>

#endif
