#include "asyffiimpl.h"

#include "absyn.h"
#include "common.h"
#include "settings.h"
#include "util.h"

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


AsyFfiRegistererImpl::AsyFfiRegistererImpl(string const& dynlibName)
    : libName(dynlibName), sym(symbol::literalTrans(dynlibName)),
      recordVar(new types::dummyRecord(sym))
{}

void AsyFfiRegistererImpl::registerFunction(
        char const* name, TAsyForeignFunction fn, AsyTypeInfo const& returnType,
        size_t numArgs, AsyFnArgMetadata* argInfoPtr
)
{
  auto* functionSig= new types::function(asyTypesEnumToTy(returnType));
  for (size_t i= 0; i < numArgs; ++i) {
    functionSig->add(asyArgInfoToFormal(argInfoPtr[i]));
  }

  recordVar->add(name, functionSig, fn);
}
record* AsyFfiRegistererImpl::getRecord() const { return recordVar; }

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
    default:
      reportError("Invalid argument type");
      return nullptr;
  }
}

ty* processArrayTypesInfoToTy(AsyTypeInfo const& asyType)
{
  auto const* typeInfo= static_cast<ArrayTypeMetadata*>(asyType.extraData);
  ty* ret= nullptr;
  switch (typeInfo->typeOfItem) {
#define CASE_ARRAY_MULTIDIM(name, dimension)                                   \
  case dimension:                                                              \
    ret= types::name##Array##dimension();                                      \
    break;

// For each type, enter another switch statement to return the correct
// type function based on the dimensions.
#define PRIMITIVE(name, Name, asyName)                                         \
  case AsyBaseTypes::Name:                                                     \
    switch (typeInfo->dimension) {                                             \
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
