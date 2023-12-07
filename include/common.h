/****
 * common.h
 *
 * Definitions common to all files.
 *****/

#ifndef COMMON_H
#define COMMON_H

#undef NDEBUG

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

#include <optional>
using std::optional;
using std::nullopt;
using std::make_optional;

using std::make_pair;

#if !defined(FOR_SHARED) &&                                             \
  ((defined(HAVE_LIBGL) && defined(HAVE_LIBGLUT) && defined(HAVE_LIBGLM)) || \
   defined(HAVE_LIBOSMESA))
#define HAVE_GL
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
#if Int_MAX2 >= 0x7fffffffffffffffLL
#define COMPACT 1
#else
#define COMPACT 0
#endif
#endif

#if COMPACT
// Reserve highest two values for DefaultValue and Undefined states.
#define Int_MAX (Int_MAX2-2)
#define int_MAX (LONG_MAX-2)
#else
#define Int_MAX Int_MAX2
#define int_MAX LONG_MAX
#endif

#define int_MIN LONG_MIN

#ifndef RANDOM_MAX
#define RANDOM_MAX 0x7FFFFFFF
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

#endif
