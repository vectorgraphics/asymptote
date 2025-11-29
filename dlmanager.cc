#include "dlmanager.h"
#include "camperror.h"
#include <utility>


#ifdef _WIN32
#  include "win32helpers.h"
#else
#  include <dlfcn.h>
#endif

namespace camp
{

LoadedDynLib::LoadedDynLib(
        string const& dlPath, bool const& closingInThread,
        TDlLoadFlags const& dlLoadFlags
)
    : storedDlPath(dlPath), dlLoadFlags(dlLoadFlags),
#ifdef _WIN32
      threadedClose(closingInThread),
      dlptr(LoadLibraryExA(dlPath.c_str(), nullptr, dlLoadFlags))
#else
      reportError("!TODO: Implement this for linux");
#endif

{
  checkLibraryNotNull();
}
LoadedDynLib::~LoadedDynLib() { closeLibrary(); }
LoadedDynLib::LoadedDynLib(LoadedDynLib const& other)
    : LoadedDynLib(other.storedDlPath, other.threadedClose, other.dlLoadFlags)
{}
LoadedDynLib& LoadedDynLib::operator=(LoadedDynLib const& other)
{
  // close existing one first, if applicable
  closeLibrary();
  this->storedDlPath= other.storedDlPath;
  this->dlLoadFlags= other.dlLoadFlags;

  // LoadLibraryExA would increase the refcount of the library
#ifdef _WIN32
  this->threadedClose= other.threadedClose;
  this->dlptr= LoadLibraryExA(storedDlPath.c_str(), nullptr, dlLoadFlags);
#else
  reportError("!TODO: Implement this for linux");
#endif
  checkLibraryNotNull();

  return *this;
}
LoadedDynLib::LoadedDynLib(LoadedDynLib&& other) noexcept
    : storedDlPath(std::move(other.storedDlPath)),
      dlLoadFlags(other.dlLoadFlags), threadedClose(other.threadedClose),
      dlptr(std::exchange(other.dlptr, nullptr))
{}
LoadedDynLib& LoadedDynLib::operator=(LoadedDynLib&& other) noexcept
{
  // close existing one first, if applicable
  closeLibrary();
  std::swap(this->storedDlPath, other.storedDlPath);
  std::swap(this->dlLoadFlags, other.dlLoadFlags);
  std::swap(this->threadedClose, other.threadedClose);

  this->dlptr= std::exchange(other.dlptr, nullptr);

  return *this;
}
TProcAddress
LoadedDynLib::getRawSymAddress(char const* symbol, bool const& check) const
{
#ifdef _WIN32
  TProcAddress const ret= GetProcAddress(dlptr, symbol);

  if (check) {
    string const outMsg= "Failed to get program address from " + string(symbol);
    w32::checkResult(ret != nullptr, outMsg);
  }
#else
  reportError("!TODO: Implement this for linux");
#endif

  return ret;
}
void LoadedDynLib::closeLibrary()
{
  if (dlptr == nullptr) {
    return;
  }

  auto const copyDlPtr= std::exchange(dlptr, nullptr);
#ifdef _WIN32
  if (threadedClose) {
    FreeLibraryAndExitThread(copyDlPtr, 0);
  } else {
    FreeLibrary(copyDlPtr);
  }
#else
  reportError("!TODO: Implement this for linux");
#endif
}
void LoadedDynLib::checkLibraryNotNull() const
{
#ifdef _WIN32
  string const outMsg= "Cannot load dll from " + storedDlPath;
  w32::checkResult(dlptr != nullptr, outMsg);
#else
  reportError("!TODO: Implement this for linux");
#endif
}

namespace
{
DynlibManager dynlibManager;
}


DynlibManager* getDynlibManager() { return &dynlibManager; }

LoadedDynLib* DynlibManager::getLib(string const& dlKey, string const& dlPath)
{
#ifdef _WIN32
  TDlLoadFlags loadFlags= 0;
#else
  TDlLoadFlags loadFlags= RTLD_LAZY;
#endif

  auto [dynlib, hasValue]= loadedDls.try_emplace(
          dlKey, std::make_unique<LoadedDynLib>(dlPath, false, loadFlags)
  );

  return dynlib->second.get();
}

void DynlibManager::delLib(string const& dlKey) { loadedDls.erase(dlKey); }

LoadedDynLib* DynlibManager::getPreloadedLib(string const& dlKey) const
{
  if (auto const dynLibIt = loadedDls.find(dlKey); dynLibIt != loadedDls.end()) {
    return dynLibIt->second.get();
  } else {
    reportError("Dll key " + dlKey + " not loaded");
    return nullptr;
  }
}
}// namespace camp