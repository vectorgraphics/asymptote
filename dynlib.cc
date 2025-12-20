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

IAsyContext* getAsyContext()
{
  return &asyContext;
}
}

void loadDynLib(string const& key)
{
#ifdef _WIN32
  string dllName= key + ".dll";
#else
  string dllName= "lib" + key + ".so";
#endif
  string fileName= settings::locateFile(dllName, true);

  camp::getDynlibManager()->getLib(key, fileName);
}

typedef void (*TVoidArgsFunction)(IAsyContext*, IAsyArgs*);

// arguments are <string key>, <string fnName>, <args1>, ..., <args4>
void callFunction1(vm::stack* stack)
{
  vm::item i1 = stack->pop();
  auto const fnName = stack->pop<string>();
  auto const dlKey = stack->pop<string>();
  
  auto const* lib= camp::getDynlibManager()->getPreloadedLib(dlKey);
  auto const fn= lib->getSymAddress<TVoidArgsFunction>(fnName.c_str(), true);
  
  camp::AsyArgsImpl args;
  args.addArgs(&i1);
  
  auto* asyContext= camp::getAsyContext();

  fn(asyContext, &args);
}

void unloadLib(string const& key) { camp::getDynlibManager()->delLib(key); }
