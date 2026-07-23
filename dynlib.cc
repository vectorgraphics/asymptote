#include "dynlib.h"

#include "asyffiimpl.h"
#include "dlmanager.h"
#include "locate.h"
#include "stack.h"
#include "genv.h"

namespace camp
{
namespace
{
AsyContextImpl asyContext;
}

IAsyContext* getAsyContext() { return &asyContext; }
}// namespace camp

auto constexpr ASYFFI_REGISTER_PLUGIN_FUNCTION_NAME= "registerAsymptotePlugin";

record* loadDynLib(string const& key, string const& fileName, trans::genv* genv)
{
  camp::LoadedDynLib const* lib=
          camp::getDynlibManager()->loadLib(key, fileName);

  camp::AsyFfiRegistererImpl registerer(key, genv, camp::getAsyContext());
  auto const registerFunction= lib->getSymAddress<TAsyRegisterDynlibFn>(
          ASYFFI_REGISTER_PLUGIN_FUNCTION_NAME
  );
  registerFunction(camp::getAsyContext(), &registerer);

  return registerer.getRecord();
}

string tryGetDllPath(string const& libName)
{
  // firstly, try to get the file as is
  auto const libPathAsIs= settings::locateFile(libName, true, settings::fs::DYNAMIC_LIB_EXTENSION);
  if (!libPathAsIs.empty() && settings::fs::extension(libPathAsIs) == settings::fs::DYNAMIC_LIB_EXTENSION_DOTTED) {
    // and only process if the file ends in .so/.dll
    return libPathAsIs;
  }
  
  // otherwise, try to find <libName>.dll or lib<libName>.so
  string dllName= libName+settings::fs::DYNAMIC_LIB_EXTENSION_DOTTED;
#ifndef _WIN32
  dllName= "lib"+libName;
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
  camp::AsyStackContextImpl stackCtx(stack);

  fn(asyContext, &stackCtx, &args, hasReturn ? &returnItem : nullptr);

  if (hasReturn) {
    stack->push(returnItem);
  }
}

void unloadLib(string const& key) { camp::getDynlibManager()->delLib(key); }
