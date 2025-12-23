#pragma once
#include "common.h"
#include <memory>

#ifdef _WIN32
#  include <Windows.h>
#else
#  include "optional"
#endif

namespace camp
{

#ifdef _WIN32
typedef HMODULE TDynLib;
typedef DWORD TDlLoadFlags;
typedef FARPROC TProcAddress;
#else
typedef void* TDynLib;
typedef int TDlLoadFlags;
typedef void const* TProcAddress;
#endif

/** RAII wrapper for loaded library handles */
class LoadedDynLib
{

public:
  /**
   *
   * @param dlPath Path to the DLL/so file
   * @param closingInThread If enabled in Windows, will close with
   * FreeLibraryAndExitThread. Otherwise, the library will be destroyed with
   * FreeLibrary. This option is inert on non-windows systems.
   * @param dlLoadFlags Will be passed into LoadLibraryEx function on windows or
   * dlopen.
   */
  LoadedDynLib(
          string const& dlPath, bool const& closingInThread= false,
          TDlLoadFlags const& dlLoadFlags= 0
  );
  ~LoadedDynLib();

  LoadedDynLib(LoadedDynLib const& other);
  LoadedDynLib& operator=(LoadedDynLib const& other);

  LoadedDynLib(LoadedDynLib&& other) noexcept;
  LoadedDynLib& operator=(LoadedDynLib&& other) noexcept;

  TProcAddress
  getRawSymAddress(char const* symbol, bool const& check= true) const;

  template<typename T>
  T getSymAddress(char const* symbol, bool const& check= true) const
  {
    return reinterpret_cast<T>(getRawSymAddress(symbol, check));
  }

private:
  void closeLibrary();
  void checkLibraryNotNull() const;

  string storedDlPath;
  TDlLoadFlags dlLoadFlags= 0;
  bool threadedClose= false;
  TDynLib dlptr= nullptr;
};

class DynlibManager
{
public:
  /**
   * @return the pointer to the loaded library or
   * loads one if it has not been loaded
   */
  LoadedDynLib* getLib(string const& dlKey, string const& dlPath);

  /** Raises an error if dlKey has already been loaded */
  LoadedDynLib* loadLib(string const& dlKey, string const& dlPath);

  [[nodiscard]]
  LoadedDynLib* getPreloadedLib(string const& dlKey) const;
  void delLib(string const& dlPath);

  void closeDynLibManager();

private:
  mem::unordered_map<string, std::unique_ptr<LoadedDynLib>> loadedDls;

  /**
   * @return A pair of dynlib pointer and boolean indicating if the insertion
   * took place. In other words, the boolean value is true iff dlKey
   * has not already been loaded
   */
  std::pair<LoadedDynLib*, bool>
  tryLoadLib(string const& dlKey, string const& dlPath);
};

DynlibManager* getDynlibManager();

#ifndef _WIN32

/** Simple wrapper around dlerror(). It calls dlerror() initially to clear any
 * errors, then if any dl operations fail, a call to dlError can be made from
 * getDlErrorMsg */
class DlErrorContext
{
public:
  DlErrorContext();
  ~DlErrorContext()= default;

  DlErrorContext(DlErrorContext const&)= delete;
  DlErrorContext& operator=(DlErrorContext const&)= delete;
  DlErrorContext(DlErrorContext&&) noexcept= delete;
  DlErrorContext& operator=(DlErrorContext&&) noexcept= delete;


  [[nodiscard]]
  optional<string> getDlErrorMsg() const;
};
#endif


}// namespace camp
