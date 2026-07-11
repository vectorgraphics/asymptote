#pragma once

/**
 * @file asyffi.h
 * @brief This file forms part of the Asymptote FFI library
 * @author Jamie Selina Lindner [jamievlin@outlook.com]
 * @remark While this file is under Apache License 2.0, other files are
 * not unless if specified explicitly.
 * @license See LICENSE-APACHE.TXT
 *
 *    Copyright 2026 Jamie Selina Lindner
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
   * On compact asymptote builds, this function behaves the same as
   * {@link setRawPointer(void*)}.
   *
   * @remark Most builds (in particular, builds suppliied from asymptote
   * website) are compact and hence if one is not targeting non-compact
   * function, there is no need to use this function.
   */
  virtual void setValueWithTypeId(void* pointer, void* tyinfo)= 0;
};

typedef void* TAsyFfiCycleToken;

class IAsyCallable
{
public:
  virtual ~IAsyCallable()= default;
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

  /**
   * Pushes a new uninitialized item and return a pointer to that item.
   * This pointer can be used to set the value of the new item.
   *
   * @remark Note that this pointer may not point to a valid item after
   * an insertion, removal, or any changes to the array.
   */
  virtual IAsyItem* pushAndReturnBlankItem()= 0;

  /** Removes the last item in the array. */
  virtual void popItem()= 0;

  [[nodiscard]]
  virtual bool isCyclic() const= 0;
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

class IAsyTuple;
class IAsyTransform;
class IAsyTensionSpecifier;
class IAsyCurlSpecifier;

class IAsyPath;
class IAsySolvedKnot;
class IAsyPath3;
class IAsyRecord;

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

  virtual IAsyItem* createBlankItem()= 0;

  /**
   * Creates a new Asymptote string.
   * @param str Contents of the string. Must be a null-terminated pointer
   * @return An opaque pointer to asymptote string. This pointer can then
   * be assigned to an item
   */
  virtual void* createNewAsyString(char const* str)= 0;
  virtual void* createNewAsyStringSized(char const* str, size_t const& size)= 0;
  virtual void updateAsyString(void* asyStringPtr, char const* str)= 0;
  virtual void
  updateAsyStringSized(void* asyString, char const* str, size_t const& size)= 0;

  /**
   * Gets the length of the specified string (this length does not include
   * the null terminator).
   */
  virtual size_t getStringLength(void* asyString)= 0;

  /**
   * Copies the string to a destination, including the null terminator of
   * the string.
   *
   * @param asyString Opaque pointer to an asymptote string instance.
   * @param destination Destination to copy to. It must point to a writable
   * buffer of at least size bufferSize.
   * @param bufferSize Size of the buffer to copy to. Make sure this number
   * is at least the length of the string + 1 (for the null terminator).
   */
  virtual void
  copyString(void* asyString, char* destination, size_t bufferSize)= 0;

  virtual IAsyArray* createNewArray(size_t const& initialSize)= 0;

  virtual IAsyTransform* createNewTransform(
          double x, double y, double xx, double xy, double yx, double yy
  )= 0;

  /** Creates a transform that is the identity function, or in tuple form,
   * (0, 0, 1, 0, 0, 1). Note that this transform can have its value changed by
   * the {@link IAsyTuple::setIndexedValue } function.
   */
  virtual IAsyTransform* createNewIdentityTransform()= 0;

  virtual IAsyTuple* createPair(double x, double y)= 0;
  virtual IAsyTuple* createTriple(double x, double y, double z)= 0;

  // tension specifier functions
  // TODO: Ask John about what exactly a tension specifier is

  /** Creates a new tension specifier with in=out=val and atleast value. */
  virtual IAsyTensionSpecifier*
  createTensionSpecifierWithSameVal(double val, bool atleast)= 0;

  /** Creates a new tension specifier with specified out, in, and atleast
   * values.*/
  virtual IAsyTensionSpecifier*
  createTensionSpecifier(double out, double in, bool atleast)= 0;

  /** Creates a new curl specifier */
  virtual IAsyCurlSpecifier* createCurlSpecifier(double value, uint8_t s)= 0;

  /** Creates a new cycle token */
  virtual TAsyFfiCycleToken createCycleToken()= 0;

  /** Gets a specified asymptote setting */
  virtual IAsyItem* getSetting(char const* name)= 0;

  // path functions

  /**
   * Creates a new 2D path.
   *
   * @param n number of knots
   * @param cycles whether this path is cyclic
   * @param numSolvedKnots number of solved knots to pass in
   * @param solvedKnotsPtr pointer to an array of {@link IAsySolvedKnot*} to
   * pass to path creation. Each solved knot must be of 2D type (having pairs as
   * values instead of triples). This pointer must point
   * to numSolvedKnots number of {@link IAsySolvedKnot*} pointers.
   *
   * @remark This object is created using Asymptote's built in allocator which
   * means garbage collection will be handled automatically.
   *
   * @return pointer to the specified 2D IAsyPath
   */
  virtual IAsyPath* createAsyPath(
          int64_t n, bool cycles, size_t numSolvedKnots,
          IAsySolvedKnot const* const* solvedKnotsPtr
  )= 0;

  /**
   * Creates a new 3D path
   *
   *
   * @param n number of knots
   * @param cycles whether this path is cyclic
   * @param numSolvedKnots number of solved knots to pass in
   * @param solvedKnotsPtr pointer to an array of {@link IAsySolvedKnot*} to
   * pass to path creation. Each solved knot must be of 3D type (having triples
   * as values instead of pairs). This pointer must point to
   * numSolvedKnots number of {@link IAsySolvedKnot*} pointers.
   *
   * @remark This object is created using Asymptote's built in allocator which
   * means garbage collection will be handled automatically.
   *
   * @return pointer to the specified 3D IAsyPath3.
   */
  virtual IAsyPath3* createAsyPath3(
          int64_t n, bool cycles, size_t numSolvedKnots,
          IAsySolvedKnot const* const* solvedKnotsPtr
  )= 0;

  /**
   * Creates a new 2D solved knot.
   *
   * @param pre pre pair point. This value must point to a pair.
   * @param point the point. This value must point to a pair.
   * @param post post pair point. This value must point to a pair.
   * @param isStraight
   *
   * @remark This object is created using Asymptote's built in allocator which
   * means garbage collection will be handled automatically.
   * @return a new 2D solved knot.
   */
  virtual IAsySolvedKnot* createSolvedKnot2D(
          IAsyTuple const* pre, IAsyTuple const* point, IAsyTuple const* post,
          bool isStraight
  )= 0;

  /**
   * Creates a new 3D solved knot.
   *
   * @param pre pre triple point. This value must point to a triple.
   * @param point the point. This value must point to a triple.
   * @param post post triple point. This value must point to a triple.
   * @param isStraight
   *
   * @remark This object is created using Asymptote's built in allocator which
   * means garbage collection will be handled automatically.
   * @return a new 3D solved knot.
   */
  virtual IAsySolvedKnot* createSolvedKnot3D(
          IAsyTuple const* pre, IAsyTuple const* point, IAsyTuple const* post,
          bool isStraight
  )= 0;

  // GC functions
  /**
   * Returns true if the garbage collector is supported in this build of
   * Asymptote, false otherwise
   */
  [[nodiscard]]
  virtual bool isGcSupported() const= 0;

  /**
   * Returns the size of gc stack base in bytes.
   *
   * @remark This is used to allocate
   * memory for registering a new thread with Asymptote's garbage collector.
   * Moreover, if GC is not supported, this function returns 0.
   */
  [[nodiscard]]
  virtual size_t getGcStackBaseSize() const= 0;

  /**
   * Gets stack base for the thread that calls this function.
   *
   * @param stackBase pointer to a memory block with size obtained from
   * {@link getGcStackBaseSize} function call.
   * @return true if successful, false otherwise
   * @remark If one is creating a new thread, be sure to call this function
   * from the same thread that is going to be used, and not from the main
   * thread. If GC is not supported, this function's behavior is undefined.
   */
  virtual bool getGcStackBase(void* stackBase)= 0;

  /**
   * Register the calling thread with the garbage collector. This will allow the
   * calling thread to perform memory-related operations.
   * @param stackBase pointer to an initialized stack base (after calling {@link
   * getGcStackBase})
   * @return true if successful or if the thread is already registered, false
   * otherwise.
   * @remark If GC is not supported, this function's behavior is undefined.
   */
  virtual bool registerThreadWithGc(void* stackBase) const= 0;

  /**
   * Unregister the calling thread from the garbage collector system.
   * @remark If GC is not supported, this function's behavior is undefined.
   */
  virtual void unregisterThreadWithGc() const= 0;

  // frame functions

  [[nodiscard]]
  /**
   * Whether the Asymptote build is of simple frame variant.
   *
   * In a simple frame build, records are passed as an {@link IAsyItem} pointing
   * to an array of variables, whereas in a non-simple frame build (which is the
   * case for a standard Asymptote build), records are passed as {@link
   * IAsyVarFrame}
   */
  virtual bool isSimpleFrameBuild() const= 0;
};

// question: will we ever exceed 256 primitive types?

namespace Asy
{

/** Types of Asymptote */
enum BaseTypes : uint8_t
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

  /** Corresponds to inferred type. Do not use this when specifying arguments */
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

  /**
   * Corresponds to array. If this type is specified in {@link Asy::TypeInfo},
   * {@link Asy::TypeInfo.arrayTypeInfo} must be specified
   */
  ArrayType,

  /** Corresponds to Asymptote structs */
  Record,

  /**
   * Corresponds to Asympote function type. If this type is specified in
   * {@link Asy::TypeInfo}, {@link Asy::TypeInfo.functionTypeInfo} must be
   * filled with appropriate information
   */
  FunctionType,
};

}// namespace Asy

class IAsyTensionSpecifier
{
public:
  virtual ~IAsyTensionSpecifier()= default;

  [[nodiscard]]
  virtual double getOut() const= 0;

  [[nodiscard]]
  virtual double getIn() const= 0;

  [[nodiscard]]
  virtual bool getAtleast() const= 0;
};

constexpr uint8_t ASY_SIDE_OUT= 0;
constexpr uint8_t ASY_SIDE_IN= 1;
constexpr uint8_t ASY_SIDE_END= 2;
constexpr uint8_t ASY_SIDE_JOIN= 3;

class IAsyCurlSpecifier
{
public:
  virtual ~IAsyCurlSpecifier()= default;

  [[nodiscard]]
  virtual double getValue() const= 0;
  [[nodiscard]]
  virtual uint8_t getSideAsInt() const= 0;
};

class IAsySolvedKnot
{
public:
  virtual ~IAsySolvedKnot()= default;

  [[nodiscard]]
  virtual IAsyTuple const* getPre() const= 0;

  [[nodiscard]]
  virtual IAsyTuple const* getPoint() const= 0;

  [[nodiscard]]
  virtual IAsyTuple const* getPost() const= 0;

  [[nodiscard]]
  virtual bool isStraight() const= 0;
};

class IAsyBbox;
class IAsyBbox3;

class IAsyPath
{
public:
  virtual ~IAsyPath()= default;

  [[nodiscard]]
  virtual bool isCyclic() const= 0;

  [[nodiscard]]
  virtual double getCachedLength() const= 0;

  /** Gets the nth solved knot.
   * @remark The pointer returned here may not be stable, which means
   * after certain operations, this pointer may no longer point to a valid
   * object
   */
  [[nodiscard]]
  virtual IAsySolvedKnot const* getNodeAt(size_t index) const= 0;

  [[nodiscard]]
  virtual size_t getNodesCount() const= 0;

  [[nodiscard]]
  virtual IAsyBbox const* getBox() const= 0;

  /** Gets the times where minimum and maximum extents are attained. */
  [[nodiscard]]
  virtual IAsyBbox const* getTimes() const= 0;
};

class IAsyPath3
{
public:
  virtual ~IAsyPath3()= default;

  [[nodiscard]]
  virtual bool isCyclic() const= 0;

  [[nodiscard]]
  virtual double getCachedLength() const= 0;

  /** Gets the nth solved knot.
   * @remark The pointer returned here may not be stable, which means
   * after certain operations, this pointer may no longer point to a valid
   * object
   */
  [[nodiscard]]
  virtual IAsySolvedKnot const* getNodeAt(size_t index) const= 0;

  [[nodiscard]]
  virtual size_t getNodesCount() const= 0;

  [[nodiscard]]
  virtual IAsyBbox3 const* getBox() const= 0;

  /** Gets the times where minimum and maximum extents are attained. */
  [[nodiscard]]
  virtual IAsyBbox3 const* getTimes() const= 0;
};

/** Interface representing guides */
class IAsyGuide
{
public:
  virtual ~IAsyGuide()= default;

  /**
   * Returns a created path that represents the guide. Path is allocated
   * using the garbage collector.
   */
  virtual IAsyPath* solveGuide()= 0;

  /** Whether the path is cyclic */
  [[nodiscard]]
  virtual bool isCyclic()= 0;

  [[nodiscard]]
  virtual uint8_t getLocation() const= 0;
};

class IAsyBbox
{
public:
  virtual ~IAsyBbox()= default;

  [[nodiscard]]
  virtual double getLeft() const= 0;

  [[nodiscard]]
  virtual double getRight() const= 0;

  [[nodiscard]]
  virtual double getTop() const= 0;

  [[nodiscard]]
  virtual double getBottom() const= 0;

  [[nodiscard]]
  virtual bool isEmpty() const= 0;
};

class IAsyBbox3 : public IAsyBbox
{
public:
  ~IAsyBbox3() override= default;

  [[nodiscard]]
  virtual double getNear()= 0;

  [[nodiscard]]
  virtual double getFar()= 0;
};

namespace Asy
{

struct TypeInfo;
struct FnArgMetadata;

struct ArrayTypeMetadata {
  /** The type of the item that the array is storing. Cannot be ArrayType */
  TypeInfo const* typeOfItem;

  /** Dimensions of the array. Must be greater than zero. */
  size_t dimension;
};

/**
 * Struct to hold data about a function type. Unlike
 * {@link Asy::FunctionTypeMetadata} AsyTypeInfo must be a pointer to an
 * existing AsyTypeInfo object. This is because the former struct has
 * a dependency on {@link Asy::TypeInfo}, however AsyTypeInfo has a dependency
 * on this struct
 */
struct FunctionTypePtrRetMetadata {

  /** Pointer to a return type object for a function */
  TypeInfo const* returnType;

  /** Number of arguments */
  size_t numArgs;

  /**
   * Pointer to an AsyFnArgMetadata array which stores information about
   * each argument
   */
  FnArgMetadata const* argInfoPtr;
};

/**
 * Information about a particular asymptote type.
 */
struct TypeInfo {
  /** Base type information */
  BaseTypes baseType;

  /**
   * Pointer to additional data. For most types, this value is not used.
   * For {@link Asy::BaseTypes::ArrayType}, extraData must point to a struct of
   * {@link Asy::ArrayTypeMetadata} in arrayTypeInfo.
   */
  union
  {
    /** This is required for arrays. */
    ArrayTypeMetadata arrayTypeInfo;

    /** This is required for function types */
    FunctionTypePtrRetMetadata functionTypeInfo;

    /** This is required for records */
    IAsyRecord* recordPtr;
  } extraData;
};

/**
 * This struct is like {@link Asy::FunctionTypePtrRetMetadata} however
 * returnType is not a pointer.
 */
struct FunctionTypeMetadata {

  /** Return type of the function */
  TypeInfo returnType;

  /** Number of arguments */
  size_t numArgs;

  /** Pointer to an array about each argument's type information */
  FnArgMetadata const* argInfoPtr;
};

/**
 * Information about each argument for an asymptote external function
 */
struct FnArgMetadata {
  /** Argument type */
  TypeInfo type;

  /** Argument name */
  char const* name;

  /** Whether the argument is optional */
  bool optional;

  /** Whether the argument needs to be specified explicitly. */
  bool explicitArgs;
};

}// namespace Asy

/**
 * Context interface for any asy operations involving stack
 * (e.g., executing code in the current environment, changing variables, etc.)
 */
class IAsyStackContext
{
public:
  virtual ~IAsyStackContext()= default;

  /**
   * Calls an asymptote function. This does not pop any value out of stack
   * after call
   *
   * @param callable Function to call
   * @param numArgs number of arguments
   * @param ptrArgs pointer to an array of IAsyItem pointers
   */
  virtual void
  callVoid(IAsyCallable* callable, size_t numArgs, IAsyItem const** ptrArgs)= 0;

  /**
   * Calls an asymptote function and pops & returns the top-most item in the
   * stack
   *
   * @param callable Function to call
   * @param numArgs number of arguments
   * @param ptrArgs pointer to an array of IAsyItem pointers
   *
   * @return Pointer to the popped item of the stack that is returned from the
   * function.
   *
   * @remark Running this function on a void callable has undefined behaviour.
   */
  virtual IAsyItem* callReturning(
          IAsyCallable* callable, size_t numArgs, IAsyItem const** ptrArgs
  )= 0;

  /**
   * Calls an asymptote function and pops & returns the top-most item in the
   * stack in an existing item
   *
   * @param callable Function to call
   * @param numArgs number of arguments
   * @param ptrArgs pointer to an array of IAsyItem pointers
   * @param returnItem pointer to an item to store the returned object
   *
   * @remark Running this function on a void callable has undefined behaviour.
   */
  virtual void callReturningToExistingItem(
          IAsyCallable* callable, size_t numArgs, IAsyItem const** ptrArgs,
          IAsyItem* returnItem
  )= 0;

  /**
   * Gets a function fnName, either from top-level or in a specified, if given.
   *
   * @param module If given, the function will be retrieved from this module,
   * otherwise if this parameter is null, will get the function from top-level.
   * @param fnName name of the function.
   * @param typeInfo Type of the built-in function.
   *
   * @return Object to the built-in function, or null if such built-in cannot be
   * found.
   */
  virtual IAsyCallable*
  getBuiltin(char const* module, char const* fnName, Asy::TypeInfo typeInfo)= 0;
};

/**
 * Setter/Getter interface for Pair and Triple types and other types
 * implementing a tuple interface
 */
class IAsyTuple
{
public:
  virtual ~IAsyTuple()= default;

  /**
   * Gets the value.
   * @param index In the case of a pair or triple,
   * 0 index corresponds to the "x" value, 1 to "y" and for triple,
   * and 2 to "z". In other cases, return the element at that index as
   * specified by the type implementing IAsyTuple.
   *
   * @return Value of that element
   */
  [[nodiscard]]
  virtual double getIndexedValue(size_t const& index) const= 0;

  /**
   * Set value.
   * @param index In the case of a pair or triple,
   *  0 corresponds to the "x" value, 1 to "y" and for triple,
   *  and 2 to "z". In other cases, sets the element at that index as specified
   *  by the type implementing IAsyTuple.
   * @param val Value to set the element to
   */
  virtual void setIndexedValue(size_t const& index, double const& val)= 0;

  /** @return 2 if object is a pair, 3 if triple. In other cases, the number
   * of elements specified by the type implementing IAsyTuple.
   */
  [[nodiscard]]
  virtual size_t getTupleSize() const= 0;
};

/**
 * Function type for foreign function.
 * Function will pass (context, stackContext, args, returnItem). If the function
 * is registered as void, returnItem will be set as nullptr
 */
typedef void (*LNK_CALL TAsyForeignFunction)(
        IAsyContext*, IAsyStackContext*, IAsyArgs*, IAsyItem*
);

/**
 * An interface representing a 2D affine transform. It is a tuple
 * (x, y, xx, xy, yx, yy) where the R^2 transform function T is defined as
 * (u, v) -> (x + xx*u + xy*v, y + yx*u + yy*v)
 */
class IAsyTransform : public IAsyTuple
{
public:
  ~IAsyTransform() override= default;

  [[nodiscard]]
  virtual double getx() const= 0;

  [[nodiscard]]
  virtual double gety() const= 0;

  [[nodiscard]]
  virtual double getxx() const= 0;

  [[nodiscard]]
  virtual double getxy() const= 0;

  [[nodiscard]]
  virtual double getyy() const= 0;

  [[nodiscard]]
  virtual double getyx() const= 0;

  virtual void setFromAnotherTransform(IAsyTransform const* other)= 0;

  /** Applies transformation from in and returns out. Both in and out
   * must be pairs */
  virtual void apply(IAsyTuple* in, IAsyTuple* out)= 0;
};

/** Interface for Asymptote global environment which manages module imports */
class IAsyGlobalEnvironment
{
public:
  virtual ~IAsyGlobalEnvironment()= default;

  /**
   * Get a file module as record. If this module has already been loaded,
   * returns the existing module, otherwise loads the module from fileName
   */
  virtual IAsyRecord*
  loadFileModule(const char* moduleName, const char* fileName)= 0;

  /** Returns an existing module. If the module is not already loaded,
   * its behavior is undefined. */
  virtual IAsyRecord* loadExistingModule(const char* id)= 0;
};

class IAsyFfiRegisterer
{
public:
  virtual ~IAsyFfiRegisterer()= default;

  virtual void registerFunction(
          char const* name, TAsyForeignFunction fn,
          Asy::FunctionTypeMetadata const& fnTypeInfo
  )= 0;

  /** @return Asymptote global module import manager */
  virtual IAsyGlobalEnvironment* getGlobalEnvironment()= 0;
};

/**
 * Interface for Asymptote variable frames, which are how
 * Asymptote internally stores its structs. Variables are accessible as
 * an array where each element is the struct's variables (or functions).
 * This virtual array stores the variables in the order they are defined in the
 * original struct.
 *
 * @remark For plugin developers & Asymptote users, be careful that the changing
 * the order of variables in the struct changes their order here. This can break
 * compatibility of external code.
 */
class IAsyVarFrame
{
public:
  virtual ~IAsyVarFrame()= default;

  // TODO: Ask John about index 0 and whether we should clarify what it stores
  //       I, Jamie, unfortunately do not know what index 0 stores.
  /**
   * Gets an item from the array. This value of this item is modifiable (e.g.
   * if the item contains a pointer, one may change the pointer's value).
   *
   * @remark For setting items, use this function to get the item and change
   * the item's values.
   *
   * @remark The first field of the struct is stored at index 1,
   * while index 0 is used for internal storage.
   */
  virtual IAsyItem* getItem(size_t const& index)= 0;

  /** Gets an item from the array. The item here is not modifiable. */
  [[nodiscard]]
  virtual IAsyItem const* getItemConst(size_t const& index) const= 0;

  /** Gets size of the variable frame (i.e. the number of elements in a struct)
   */
  [[nodiscard]]
  virtual size_t getSize() const= 0;

  /**
   * Extends the variable frame to fit n elements. This function is inert
   * if the variable frame already holds more n elements or more.
   */
  virtual void extend(size_t const& n)= 0;
};

/**
 * Interface for Proto-Environment.
 * This is where Asymptote stores type definitions and variable names.
 *
 */
class IAsyProtoEnvironment
{
public:
  virtual ~IAsyProtoEnvironment()= default;

  /** Return the type associated with the typename.
   * If the type does not exist, returns a null pointer. */
  virtual void* getType(char const* typeName)= 0;

  /** Return the type as record.
   * If the type does not exist or exists but is not a record (struct) type,
   * this function returns a nullptr. */
  virtual IAsyRecord* getTypeAsRecord(char const* typeName)= 0;
};

/**
 * Interface for Asymptote records. Records are how Asymptote stores internal
 * imported files, code blocks and struct definitions. Note that IAsyRecord do
 * not store individual instances of structs, but rather definitions (e.g. which
 * fields are present, which code should get executed).
 *
 * To access a data of a struct instance, use {@link IAsyVarFrame}.
 */
class IAsyRecord
{
public:
  virtual ~IAsyRecord()= default;

  /** Return the proto-environment (type & variable records) associated with
   * this struct/code block */
  virtual IAsyProtoEnvironment* getProtoEnvironment()= 0;

  /** Return the post-definition proto-environment (type & variable records)
   * associated with this struct/code block */
  virtual IAsyProtoEnvironment* getPostDefinitionProtoEnvironment()= 0;
};

typedef void (*LNK_CALL TAsyRegisterDynlibFn)(IAsyContext*, IAsyFfiRegisterer*);

// convenience macros for asy function registration

#define REGISTER_FN_NAME(libname) registerPlugin_##libname

#define REGISTER_FN_SIG(libname)                                               \
  void LNK_CALL REGISTER_FN_NAME(libname)(                                     \
          IAsyContext * context, IAsyFfiRegisterer * registerer                \
  )

#define DECLARE_REGISTER_FN(libname)                                           \
  extern "C" ASY_FFI_EXPORT REGISTER_FN_SIG(libname)

#define ASY_FOREIGN_FUNC_SIG(functionName)                                     \
  void LNK_CALL functionName(                                                  \
          IAsyContext* context, IAsyStackContext* stackContext,                \
          IAsyArgs* args, IAsyItem* returnValue                                \
  )

/**
 * Shorthand for "functionName", &functionName for registering functions
 */
#define ASYFFI_FN_NAME_AND_ADDR(functionName) #functionName, &(functionName)
