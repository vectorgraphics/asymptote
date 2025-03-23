/**
 * @file tinyexr_impl.cc
 * @brief Implementation file for tinyexr
 */

#if defined(HAVE_CONFIG_H)
#include "config.h"
#endif

// zlib
#if defined(HAVE_ZLIB)
#define TINYEXR_USE_MINIZ 0
#include <zlib.h>
#else
#define TINYEXR_USE_MINIZ 1
#endif

#define TINYEXR_USE_THREAD 0

#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>
