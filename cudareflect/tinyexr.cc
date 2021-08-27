/**
* @file tinyexr.cc
* @brief An implementation object file for tinyexr mandated by the repository.
* 
* On Windows, use vcpkg to install zlib instead of nuget. On Linux, this should work naturally
*/

#include <zlib.h>

#define TINYEXR_IMPLEMENTATION
#define TINYEXR_USE_MINIZ 0
#define TINYEXR_USE_THREAD 1
#define TINYEXR_USE_PIZ 1

#include "tinyexr/tinyexr.h"