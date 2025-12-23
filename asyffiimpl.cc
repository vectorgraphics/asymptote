#include "asyffiimpl.h"

#include "absyn.h"
#include "common.h"
#include "util.h"

namespace camp
{

size_t AsyArgsImpl::getArgumentCount() const { return argsStorage.size(); }


IAsyItem* AsyArgsImpl::getNumberedArg(const size_t& argNum) const
{
  return argsStorage.at(argNum);
}


void AsyArgsImpl::addArgs(IAsyItem* arg) { argsStorage.push_back(arg); }
void* AsyContextImpl::malloc(size_t const& size) { return asy_malloc(size); }
void* AsyContextImpl::mallocAtomic(size_t const& size)
{
  return asy_malloc_atomic(size);
}

AsyFfiRegistererImpl::AsyFfiRegistererImpl(string const& dynlibName)
    : libName(dynlibName), sym(symbol::literalTrans(dynlibName)),
      recordVar(new types::dummyRecord(sym))
{}

void AsyFfiRegistererImpl::registerFunction(
        char const* name, TAsyForeignFunction fn, AsyTypes const& returnType,
        size_t numArgs, AsyFnArgMetadata* argInfoPtr
)
{
  auto* functionSig= new types::function(types::primVoid());
  for (int i= 0; i < numArgs; ++i) {
    auto* argInfo= argInfoPtr + i;
    functionSig->add(asyArgInfoToFormal(*argInfo));
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
