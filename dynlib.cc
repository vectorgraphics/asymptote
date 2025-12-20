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

void callFunction(
        string const& key, string const& fnName, camp::AsyArgsImpl* args
)
{
  auto const* lib= camp::getDynlibManager()->getPreloadedLib(key);
  auto const fn= lib->getSymAddress<TVoidArgsFunction>(fnName.c_str(), true);
  auto* asyContext= camp::getAsyContext();

  fn(asyContext, &args);
}

void unloadLib(string const& key) { camp::getDynlibManager()->delLib(key); }
