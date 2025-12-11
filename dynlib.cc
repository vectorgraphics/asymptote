#include "dynlib.h"

#include "dlmanager.h"
#include "locate.h"

void loadDynLib(string const& key)
{
#ifdef _WIN32
  string dllName= key + ".dll";
#else
  string dllName= "lib" + key + ".so";
#endif
  string fileName= settings::locateFile(dllName);

  camp::getDynlibManager()->getLib(key, fileName);
}

typedef void (*TVoidVoidFunction)();

void callFunction(string const& key, string const& fnName)
{
  auto const* lib= camp::getDynlibManager()->getPreloadedLib(key);
  auto const fn= lib->getSymAddress<TVoidVoidFunction>(fnName.c_str(), true);

  fn();
}

void unloadLib(string const& key) { camp::getDynlibManager()->delLib(key); }
