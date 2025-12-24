#include "dynlib.h"

#include "asyffiimpl.h"
#include "dlmanager.h"
#include "locate.h"
#include "stack.h"

namespace camp
{
namespace
{
AsyContextImpl asyContext;
}

IAsyContext* getAsyContext() { return &asyContext; }
}// namespace camp

record* loadDynLib(string const& key, string const& fileName)
{
  camp::LoadedDynLib const* lib=
          camp::getDynlibManager()->loadLib(key, fileName);
  string const registerName= string("registerPlugin_") + key;

  camp::AsyFfiRegistererImpl registerer(key);
  auto const registerFunction=
          lib->getSymAddress<TAsyRegisterDynlibFn>(registerName.c_str());
  registerFunction(camp::getAsyContext(), &registerer);

  return registerer.getRecord();
}

string tryGetDllPath(string const& libName)
{
#ifdef _WIN32
  string const dllName= libName + ".dll";
#else
  string const dllName= "lib" + libName + ".so";
#endif
  return settings::locateFile(dllName, true);
}

void callForeignFunction(vm::stack* stack, TAsyForeignFunction const fn)
{
  bool const hasReturn= stack->pop<Int>() != 0;
  auto const numArgs= stack->pop<Int>();

  camp::AsyArgsImpl args(numArgs);
  vm::item returnItem;// not always used;

  for (Int i= numArgs - 1; i >= 0; --i) {
    // stack pop is in reverse order, hence we need to
    // set the args in reverse order

    // TODO: Can we optimize this somehow?
    args.setArgNum(i, stack->pop());
  }

  auto* asyContext= camp::getAsyContext();

  fn(asyContext, &args, hasReturn ? &returnItem : nullptr);

  if (hasReturn) {
    stack->push(returnItem);
  }
}

void unloadLib(string const& key) { camp::getDynlibManager()->delLib(key); }
