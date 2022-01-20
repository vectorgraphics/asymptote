/**
* @file tinyexr.cc
* @brief An implementation object file for tinyexr mandated by the repository.
*
* On Windows, use vcpkg to install zlib instead of nuget. On Linux, this should work naturally
*/

#include <zlib.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0

#ifdef HAVE_PTHREAD
#define TINYEXR_USE_THREAD 1
#else
#define TINYEXR_USE_THREAD 0
#endif

#include "cudareflect/tinyexr/tinyexr.h"
