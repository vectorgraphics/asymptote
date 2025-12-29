#include "asyffiimpl.h"

#include "absyn.h"
#include "common.h"
#include "settings.h"
#include "util.h"

namespace camp
{

AsyArgsImpl::AsyArgsImpl(size_t const& argSize)
  : argsStorage(argSize)
{

}
void AsyArgsImpl::setArgNum(size_t const& argNum, vm::item const& arg)
{
  argsStorage[argNum] = arg;
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
        char const* name, TAsyForeignFunction fn, AsyTypes const& returnType,
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

ty* asyTypesEnumToTy(AsyTypes const& asyType)
{
  switch (asyType) {
    case Void:
      return types::primVoid();
    case Integer:
      return types::primInt();
    case Real:
      return types::primReal();
    case Pair:
      return types::primPair();
    case Triple:
      return types::primTriple();
    default:
      reportError("Invalid argument type");
      return nullptr;
  }
}

types::formal asyArgInfoToFormal(AsyFnArgMetadata const& argInfo)
{
  ty* parsedArgType= asyTypesEnumToTy(argInfo.type);
  return {parsedArgType, symbol::literalTrans(argInfo.name), argInfo.optional,
          argInfo.explicitArgs};
}


}// namespace camp
