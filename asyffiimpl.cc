#include "asyffiimpl.h"

#include "absyn.h"
#include "common.h"
#include "settings.h"
#include "transform.h"
#include "util.h"

#include "triple.h"
#include <array.h>
#include <callable.h>

#include <guide.h>
#include <stack.h>

namespace camp
{

AsyArgsImpl::AsyArgsImpl(size_t const& argSize) : argsStorage(argSize) {}
void AsyArgsImpl::setArgNum(size_t const& argNum, vm::item const& arg)
{
  argsStorage[argNum]= arg;
}

size_t AsyArgsImpl::getArgumentCount() const { return argsStorage.size(); }


IAsyItem* AsyArgsImpl::getNumberedArg(const size_t& argNum)
{
  return argsStorage.data() + argNum;
}


void* AsyContextImpl::malloc(size_t const& size) { return asy_malloc(size); }
void* AsyContextImpl::mallocAtomic(size_t const& size)
{
  return asy_malloc_atomic(size);
}
bool AsyContextImpl::isCompactBuild() const
{
#if COMPACT
  return true;
#else
  return false;
#endif
}
const char* AsyContextImpl::getVersion() const { return REVISION; }
const char* AsyContextImpl::getAsyGlVersion() const { return AsyGLVersion; }

IAsyItem* AsyContextImpl::createBlankItem() { return new vm::item(); }

void* AsyContextImpl::createNewAsyString(char const* str)
{
  return new (UseGC) mem::string(str);
}

void* AsyContextImpl::createNewAsyStringSized(
        char const* str, size_t const& size
)
{
  return new (UseGC) mem::string(str, size);
}
void AsyContextImpl::updateAsyString(void* asyStringPtr, const char* str)
{
  auto* castedStr= static_cast<mem::string*>(asyStringPtr);
  castedStr->assign(str);
}
void AsyContextImpl::updateAsyStringSized(
        void* asyStringPtr, const char* str, const size_t& size
)
{
  auto* castedStr= static_cast<mem::string*>(asyStringPtr);
  castedStr->assign(str, size);
}
IAsyArray* AsyContextImpl::createNewArray(const size_t& initialSize)
{
  return new vm::array(initialSize);
}
IAsyTransform* AsyContextImpl::createNewTransform(
        double x, double y, double xx, double xy, double yx, double yy
)
{
  return createNewItemGeneric<transform, IAsyTransform>(x, y, xx, xy, yx, yy);
}
IAsyTransform* AsyContextImpl::createNewIdentityTransform()
{
  return createNewTransform(0, 0, 1, 0, 0, 1);
}
IAsyTuple* AsyContextImpl::createPair(double x, double y)
{
  return createNewItemGeneric<pair, IAsyTuple>(x, y);
}
IAsyTuple* AsyContextImpl::createTriple(double x, double y, double z)
{
  return createNewItemGeneric<triple, IAsyTuple>(x, y, z);
}

IAsyTensionSpecifier*
AsyContextImpl::createTensionSpecifierWithSameVal(double val, bool atleast)
{
  return new tensionSpecifier(val, atleast);
}
IAsyTensionSpecifier*
AsyContextImpl::createTensionSpecifier(double out, double in, bool atleast)
{
  return new tensionSpecifier(out, in, atleast);
}
IAsyCurlSpecifier* AsyContextImpl::createCurlSpecifier(double value, uint8_t s)
{
  return new curlSpecifier(value, static_cast<side>(s));
}
TAsyFfiCycleToken AsyContextImpl::createCycleToken() { return new cycleToken; }

AsyStackContextImpl::AsyStackContextImpl(vm::stack* inStack) : stack(inStack) {}
void AsyStackContextImpl::callVoid(
        IAsyCallable* callable, size_t const numArgs, IAsyItem const** ptrArgs
)
{
  for (size_t i= 0; i < numArgs; ++i) {
    auto* ptrArg= dynamic_cast<vm::item const*>(ptrArgs[i]);
    if (!ptrArg) {
      reportError("Invalid item supplied as an argument");
      return;
    }
    stack->push(*ptrArg);
  }

  auto* fn= dynamic_cast<vm::callable*>(callable);
  if (!fn) {
    reportError("Invalid function supplied");
    return;
  }

  fn->call(stack);
}

IAsyItem* AsyStackContextImpl::callReturning(
        IAsyCallable* callable, size_t const numArgs, IAsyItem const** ptrArgs
)
{
  callVoid(callable, numArgs, ptrArgs);
  return new vm::item(stack->pop());
}
void AsyStackContextImpl::callReturningToExistingItem(
        IAsyCallable* callable, size_t numArgs, const IAsyItem** ptrArgs,
        IAsyItem* returnItem
)
{
  callVoid(callable, numArgs, ptrArgs);

  auto* retItemCasted= dynamic_cast<vm::item*>(returnItem);
  if (!retItemCasted) {
    reportError("Invalid return object specified");
    return;
  }
  *retItemCasted= stack->pop();
}


AsyFfiRegistererImpl::AsyFfiRegistererImpl(string const& dynlibName)
    : libName(dynlibName), sym(symbol::literalTrans(dynlibName)),
      recordVar(new types::dummyRecord(sym))
{}

void AsyFfiRegistererImpl::registerFunction(
        char const* name, TAsyForeignFunction fn,
        AsyFunctionTypeMetadata const& fnTypeInfo
)
{
  AsyFunctionTypePtrRetMetadata const fnMetadataPtr= {
          &(fnTypeInfo.returnType), fnTypeInfo.numArgs, fnTypeInfo.argInfoPtr
  };

  types::function* functionSig= createFunctionTypeFromMetadata(fnMetadataPtr);
  recordVar->add(name, functionSig, fn);
}
record* AsyFfiRegistererImpl::getRecord() const { return recordVar; }

types::function*
createFunctionTypeFromMetadata(AsyFunctionTypePtrRetMetadata const& fnTypeInfo)
{
  auto* functionSig=
          new types::function(asyTypesEnumToTy(*(fnTypeInfo.returnType)));
  for (size_t i= 0; i < fnTypeInfo.numArgs; ++i) {
    functionSig->add(asyArgInfoToFormal(fnTypeInfo.argInfoPtr[i]));
  }

  return functionSig;
}

ty* asyTypesEnumToTy(AsyTypeInfo const& asyType)
{
  switch (asyType.baseType) {
#define PRIMITIVE(name, Name, asyName)                                         \
  case AsyBaseTypes::Name:                                                     \
    return types::prim##Name();
#define EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#define PRIMITIVES_MACRO_ONLY
#include "primitives.h"


    DEFINE_PRIMTIVES
#undef EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#undef PRIMITIVES_MACRO_ONLY
#undef PRIMITIVE
    case Integer:// handle integer case separately
      return types::primInt();
    case Str:
      return types::primString();
    case ArrayType:
      return processArrayTypesInfoToTy(asyType);
    case FunctionType:
      return createFunctionTypeFromMetadata(asyType.extraData.functionTypeInfo);
    default:
      reportError("Invalid argument type");
      return nullptr;
  }
}

ty* processArrayTypesInfoToTy(AsyTypeInfo const& asyType)
{
  auto const& typeInfo= asyType.extraData.arrayTypeInfo;
  ty* ret= nullptr;
  switch (typeInfo.typeOfItem) {
#define CASE_ARRAY_MULTIDIM(name, dimension)                                   \
  case dimension:                                                              \
    ret= types::name##Array##dimension();                                      \
    break;

// For each type, enter another switch statement to return the correct
// type function based on the dimensions.
#define PRIMITIVE(name, Name, asyName)                                         \
  case AsyBaseTypes::Name:                                                     \
    switch (typeInfo.dimension) {                                              \
      case 1:                                                                  \
        ret= types::name##Array();                                             \
        break;                                                                 \
        CASE_ARRAY_MULTIDIM(name, 2)                                           \
        CASE_ARRAY_MULTIDIM(name, 3)                                           \
      default:                                                                 \
        break;                                                                 \
    }                                                                          \
    break;
#define EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#define PRIMITIVES_MACRO_ONLY
#include "primitives.h"


    DEFINE_PRIMTIVES
    PRIMITIVE(Int, Integer, _)
    PRIMITIVE(string, Str, _)
    default:
      break;
#undef EXCLUDE_POTENTIALLY_CONFLICTING_NAME_TYPE
#undef PRIMITIVES_MACRO_ONLY
#undef PRIMITIVE
  }

  if (ret == nullptr) {
    reportError("Invalid dimensons or type information");
  }
  return ret;
}

types::formal asyArgInfoToFormal(AsyFnArgMetadata const& argInfo)
{
  ty* parsedArgType= asyTypesEnumToTy(argInfo.type);
  return {parsedArgType, symbol::literalTrans(argInfo.name), argInfo.optional,
          argInfo.explicitArgs};
}


}// namespace camp
