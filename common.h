/****
 * common.h
 *
 * Definitions common to all files.
 *****/

#ifndef COMMON_H
#define COMMON_H

#undef NDEBUG

#if defined(_WIN32)
#include <Winsock2.h>
#endif

#include <iostream>
#include <memory>
#include <climits>
#include <cstdint>

#if defined(_MSC_VER)
// for and/or/not operators. not supported natively on MSVC
#include <BaseTsd.h>
#include <ciso646>
typedef SSIZE_T ssize_t;
#define STDOUT_FILENO 1
#define STDIN_FILENO 0
#define STDERR_FILENO 2
#endif

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef DEBUG_CSTACKTRACE
// To output a stacktrace when you are compiling with
// CFLAGS="-g -O0 -DDEBUG_CSTACKTRACE", insert code like the following:
// cerr << boost::stacktrace::stacktrace();
// at the appropriate place.
// NOTE: This gives useful information on MacOS, but seems to give only
// hex addresses (rather than function names) on Linux. Possible remedies
// may be found at https://stackoverflow.com/q/52583544/2318074 and linked
// pages.
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#  endif

#  include <boost/stacktrace.hpp>
#endif

#ifndef OMIT_OPTIONAL
#include "optional.hpp"
using nonstd::optional;
using nonstd::nullopt;
using nonstd::make_optional;
#endif

using std::make_pair;

#if !defined(FOR_SHARED) && defined(HAVE_LIBVULKAN) && defined(HAVE_LIBGLM)
 #define HAVE_VULKAN
 #endif

#if defined(HAVE_LIBREADLINE) || defined(HAVE_LIBEDIT)
#define HAVE_READLINE
#endif

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

#include "memory.h"

#define Int_MAX2 INT64_MAX
#define Int_MIN INT64_MIN
typedef int64_t Int;
typedef uint64_t unsignedInt;

#ifndef COMPACT
#define COMPACT 1
#endif

#if COMPACT
// Reserve highest two values for DefaultValue and Undefined states.
#define Int_MAX (Int_MAX2-2)
#else
#define Int_MAX Int_MAX2
#endif

using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::istream;
using std::ostream;

using mem::string;
using mem::stringstream;
using mem::istringstream;
using mem::ostringstream;
using mem::stringbuf;

using std::shared_ptr;
using std::unique_ptr;
using std::make_shared;

static const struct ws_t {} ws={};

// Portable way of skipping whitespace
inline std::istream &operator >> (std::istream & s, const ws_t &ws) {
  if(!s.eof())
    s >> std::ws;
  return s;
}

#if defined(_WIN32)
#include "win32helpers.h"
#define unlink _unlink
#endif

#endif
