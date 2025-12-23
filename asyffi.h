#pragma once

/**
 * @file asyffi.h
 * @brief This file forms part of the Asymptote FFI library
 */

#include <cstdint>

#if defined(_WIN32)
#  define ASY_FFI_EXPORT __declspec(dllexport)

// In case someone targets 32-bit x86 systems, this calling convention is
// needed
#ifndef _M_X64
#define LNK_CALL __cdecl
#else
#define LNK_CALL
#endif

#elif defined(__linux__)
#  include <cstddef>
#  define ASY_FFI_EXPORT [[gnu::visibility("default")]]

// like the case for windows. LNK_CALL is not used for 64-bit systems
#ifndef __LP64__
#define LNK_CALL __attribute__((__cdecl__))
#else
#define LNK_CALL
#endif

#elif defined(__APPLE__)

// As Jamie does not have a Mac machine, if FFI is to be supported on macs,
// someone else has to implement it ( unfortunately :( )
#error FFI is not yet ready for mac-based systems. \
  Additionally, work has to be done for ARM vs x86 systems
#else
#error Right now, FFI is only supported for windows and linux systems
#endif

class IAsyItem
{
public:
  virtual ~IAsyItem()= default;

  [[nodiscard]]
  virtual int64_t asInt64() const= 0;

  [[nodiscard]]
  virtual double asDouble() const= 0;

  [[nodiscard]]
  virtual void* asRawPointer() const= 0;
};

/** Interface for asymptote arguments */
class IAsyArgs
{
public:
  virtual ~IAsyArgs()= default;

  [[nodiscard]]
  virtual size_t getArgumentCount() const= 0;

  [[nodiscard]]
  virtual double getNumberedArgAsReal(size_t const& argNum) const= 0;

  [[nodiscard]]
  virtual int64_t getNumberedArgAsInt(size_t const& argNum) const= 0;
};
