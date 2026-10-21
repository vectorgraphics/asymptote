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
  // Resolve the exact name the user typed, with the platform dynamic-library
  // extension (.so on UNIX, .dll on Windows), across every Asy search path.
  // The loader adds and strips NO 'lib' prefix: import a file named libfoo.so
  // as `import libfoo;`. (The CMake helper names output lib<name> on all
  // platforms; that is a build-side detail the import machinery does not parse.)
  string const path=
      settings::locateFile(libName, true, settings::fs::DYNAMIC_LIB_EXTENSION);
  // Accept only a file that really carries the platform extension, so
  // `import X;` resolves to X.so/X.dll and never a bare extension-less X.
  if (settings::fs::extension(path) == settings::fs::DYNAMIC_LIB_EXTENSION_DOTTED)
    return path;
  return string();
}

void callForeignFunction(vm::stack* stack, TAsyForeignFunction const fn, string const& fnName)
{
  bool const hasReturn= stack->pop<Int>() != 0;
  auto const numArgs= stack->pop<Int>();

  camp::AsyArgsImpl args(numArgs, fnName);
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

void ffiNotSupportedError(string const& dllPath)
{
  em.sync();
  em << "error: FFI is not enabled. Cannot load '" << dllPath << "'\n";
  em.sync(true);
  throw handled_error();
}
