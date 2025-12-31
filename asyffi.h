#pragma once

/**
 * @file asyffi.h
 * @brief This file forms part of the Asymptote FFI library
 */

#include <cstdint>
#include <typeinfo>

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

#  error FFI is not yet implementy for MacOS. \
  Additionally, work has to be done for ARM vs x86 systems.
#else
#  error Right now, FFI is only supported for MSWindows and Linux systems
#endif

/**
 * These values are used for compact builds of asymptote
 * (most builds by default)
 */
constexpr int64_t ASY_COMPACT_DEFAULT_VALUE= 0x7fffffffffffffffLL;
constexpr int64_t ASY_COMPACT_UNDEFINED_VALUE= 0x7ffffffffffffffeLL;
constexpr int64_t ASY_COMPACT_BOOL_TRUTH_VALUE= 0xABABABABABABABACLL;
constexpr int64_t ASY_COMPACT_BOOL_FALSE_VALUE= 0xABABABABABABABABLL;

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
  virtual bool asBoolean() const= 0;

  [[nodiscard]]
  virtual bool isDefault() const= 0;

  virtual void setInt64Value(int64_t const& value)= 0;
  virtual void setDoubleValue(double const& value)= 0;
  virtual void setRawPointer(void* pointer)= 0;
  virtual void setBooleanValue(bool const& value)= 0;

  /**
   * Sets value with type info. This method is only relevant if
   * the asymptote host is of a non-compact build.
   *
   * On compact asymptote builds, this function behave the same as
   * {@link setRawPointer(void*)}.
   *
   * @remark Most builds (in particular, builds suppliied from asymptote
   * website) are compact and hence if one is not targeting non-compact
   * function, there is no need to use this function.
   */
  virtual void setValueWithTypeId(void* pointer, std::type_info* tyinfo)= 0;
};

/** Interface for Asymptote array. */
class IAsyArray
{
public:
  virtual ~IAsyArray()= default;

  /**
   * Gets an item at the specified position.
   * Attempting to get an item at a position beyond the array's size
   * has undefined behavior.
   */
  [[nodiscard]]
  virtual IAsyItem* getItem(size_t const& position)= 0;

  /**
   * Sets an item at the specified position.
   * Attempting to set an item at a position beyond the array's size
   * has undefined behavior. Note that the item is copied, hence there is no
   * need to preserve the itemToSet pointer.
   */
  virtual void setItem(size_t const& position, IAsyItem* itemToSet)= 0;

  /** Gets the size of an array */
  [[nodiscard]]
  virtual size_t getSize() const= 0;

  /** Resizes the array. New values may not be initialized */
  virtual void setSize(size_t const& newSize)= 0;

  /**
   * Pushes itemToAdd to the end of an array. Note that the item is copied
   * so there is no need to preserve the itemToAdd's object.
   */
  virtual void pushItem(IAsyItem* itemToAdd)= 0;

  /** Removes the last item in the array. */
  virtual void popItem()=0;

  [[nodiscard]]
  virtual bool isCyclic() const=0;
  virtual void setCyclic(bool const& isCyclic)= 0;
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

  /**
   * In a compact build, the couple highest values in Int (INT_MAX) are
   * reserved for default/undefined values. In addition, there are speical
   * numbers to store "truthy" and "falsy" values for boolean
   */
  [[nodiscard]]
  virtual bool isCompactBuild() const= 0;

  [[nodiscard]]
  virtual char const* getVersion() const= 0;

  [[nodiscard]]
  virtual char const* getAsyGlVersion() const= 0;
};

// question: will we ever exceed 256 primitive types?

/** Types of Asymptote */
enum AsyBaseTypes : uint8_t
{
  /** Corresponds to void.
   * If used as function return type, will not return any value */
  Void= 0,

  /** Corresponds to real (double-precision floating point) */
  Real,

  /** Corresponds to int (64-bit integer) */
  Integer,

  /** Corresponds to pair (x,y), where x and y are real values */
  Pair,
  /** Corresponds to triple (x,y,z), where x, y, and z are real values */
  Triple,

  /** Corresponds to bool */
  Boolean,

  /** Corresponds to string */
  Str,

  /** Corresponds to inferred type */
  Inferred,

  /** Corresponds to 2D affine transform type */
  Transform,

  /** Corresponds to guide */
  Guide,

  /** Corresponds to path in 2D space */
  Path,

  /** Corresponds to path in 3D space */
  Path3,

  /** Corresponds to cycle token */
  CycleToken,

  /** Corresponds to tension specifier for paths */
  TensionSpecifier,

  /** Corresponds to curl specifier for paths */
  CurlSpecifier,

  /** Corresponds to Pen */
  Pen,

  /** Corresponds to picture */
  Picture,

  /** Corresponds to file */
  File,

  /** Corresponds to code */
  Code,

  /** Corresponds to array */
  ArrayType,

  /** Corresponds to Asymptote structs */
  Record,
};

struct AsyTypeInfo {
  AsyBaseTypes baseType;

  /**
   * Pointer to additional data. For most types, this value is not used.
   * For {@link AsyBaseTypes::ArrayType}, extraData must point to a struct of
   * {@link AsyArrayTypeMetadata}.
   */
  void* extraData;
};

struct AsyArrayTypeMetadata {
  /** The type of the item that the array is storing. Cannot be array*/
  AsyBaseTypes typeOfItem;

  /** Dimensions, Can be 1, 2, or 3. */
  size_t dimension;

  /** Currently unused. May be used in the future */
  void* extraData;
};

struct AsyFnArgMetadata {
  AsyTypeInfo type;
  char const* name;
  bool optional;
  bool explicitArgs;

  // to be used in the future?
  void* extraData;
};

/** Setter/Getter interface for Pair and Triple types */
class IAsyDoubleTuple
{
public:
  virtual ~IAsyDoubleTuple()= default;

  /**
   * Gets the value.
   * @param index 0 index corresponds to the "x" value,
   * 1 to "y" and for triple, and 2 to "z".
   * Any other value will cause an error
   *
   * @return Value of that element
   */
  [[nodiscard]]
  virtual double getIndexedValue(size_t const& index) const= 0;

  /**
   * Set value.
   * @param index 0 index corresponds to the "x" value, 1 to "y" and for triple,
   *  and 2 to "z". Any other value will cause an error
   * @param val Value to set the element to
   */
  virtual void setIndexedValue(size_t const& index, double const& val)= 0;

  /** @return 2 if object is a pair, 3 if triple */
  virtual size_t getTupleSize() const= 0;
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
          char const* name, TAsyForeignFunction fn,
          AsyTypeInfo const& returnType, size_t numArgs,
          AsyFnArgMetadata* argInfoPtr
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

/**
 * Shorthand for "functionName", &functionName for registering functions
 */
#define ASYFFI_FN_NAME_AND_ADDR(functionName) #functionName, &(functionName)
