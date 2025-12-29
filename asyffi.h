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
#  ifndef _M_X64
#    define LNK_CALL __cdecl
#  else
#    define LNK_CALL
#  endif

#elif defined(__linux__)
#  include <cstddef>
#  define ASY_FFI_EXPORT [[gnu::visibility("default")]]

// like the case for windows. LNK_CALL is not used for 64-bit systems
#  ifndef __LP64__
#    define LNK_CALL __attribute__((__cdecl__))
#  else
#    define LNK_CALL
#  endif

#elif defined(__APPLE__)

// As Jamie does not have a Mac machine, if FFI is to be supported on Macs,
// someone else has to implement it (unfortunately :( )
#  error FFI is not yet ready for mac-based systems. \
  Additionally, work has to be done for ARM vs x86 systems
#else
#  error Right now, FFI is only supported for windows and linux systems
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

  [[nodiscard]]
  virtual bool isDefault() const= 0;

  virtual void setInt64Value(int64_t const& value)= 0;
  virtual void setDoubleValue(double const& value)= 0;
  virtual void setRawPointer(void* pointer)= 0;
};

/** Interface for asymptote arguments */
class IAsyArgs
{
public:
  virtual ~IAsyArgs()= default;

  [[nodiscard]]
  virtual size_t getArgumentCount() const= 0;

  [[nodiscard]]
  virtual IAsyItem* getNumberedArg(size_t const& argNum)= 0;
};

class IAsyContext
{
public:
  virtual ~IAsyContext()= default;

  virtual void* malloc(size_t const& size)= 0;
  virtual void* mallocAtomic(size_t const& size)= 0;
};

// question: will we ever exceed 256 primitive types?

/** Types of Asymptote */
enum AsyTypes : uint8_t
{
  /** Corresponds to the void type.
   * If used as function return type, will not return any value */
  Void= 0,

  /** Corresponds to asy real type, or double-precision floating point*/
  Real= 1,

  /** Corresponds to Int, or 64-bit integer*/
  Integer= 2,

  /** Corresponds to Pair (x, y), where x and y are real values */
  Pair= 3,
  /** Corresponds to Triple (x, y, z) where x, y, and z are real values */
  Triple= 4,
  // TODO: Add more types to this
};

struct AsyFnArgMetadata {
  AsyTypes type;
  char const* name;
  bool optional;
  bool explicitArgs;

  // to be used in the future?
  void* extraData;
};

/**
 * Function type for foreign function.
 * Function will pass (context, args, returnItem). If the function is
 * registered as void, returnItem will be set as nullptr
 */
typedef void (*LNK_CALL TAsyForeignFunction)(
        IAsyContext*, IAsyArgs*, IAsyItem*
);

class IAsyFfiRegisterer
{
public:
  virtual ~IAsyFfiRegisterer()= default;

  virtual void registerFunction(
          char const* name, TAsyForeignFunction fn, AsyTypes const& returnType,
          size_t numArgs, AsyFnArgMetadata* argInfoPtr
  )= 0;
};

typedef void (*LNK_CALL TAsyRegisterDynlibFn)(IAsyContext*, IAsyFfiRegisterer*);

#define REGISTER_FN_NAME(libname) registerPlugin_##libname

#define REGISTER_FN_SIG(libname)                                               \
  void LNK_CALL REGISTER_FN_NAME(libname)(                                     \
          IAsyContext * context, IAsyFfiRegisterer * registerer                \
  )

#define DECLARE_REGISTER_FN(libname)                                           \
  extern "C" ASY_FFI_EXPORT REGISTER_FN_SIG(libname)

#define ASY_FOREIGN_FUNC_SIG(functionName)                                     \
  void LNK_CALL functionName(                                                  \
          IAsyContext* context, IAsyArgs* args, IAsyItem* returnValue          \
  )
