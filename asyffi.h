#pragma once

/**
 * @file asyffi.h
 * @brief This file forms part of the Asymptote FFI library
 */

#include <cstdint>

#ifdef _WIN32
#  define ASY_FFI_EXPORT __declspec(dllexport)
#else
#include <cstddef>
#  define ASY_FFI_EXPORT [[gnu::visibility("default")]]
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
