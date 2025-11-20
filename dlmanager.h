#pragma once
#include "common.h"

#ifdef _WIN32
#  include <Windows.h>
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
typedef void* TProcAddress;
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
  T* getSymAddress(char const* symbol, bool const& check= true) const
  {
    return reinterpret_cast<T*>(getRawSymAddress(symbol, check));
  }

private:
  void closeLibrary();
  void checkLibraryNotNull() const;

  string storedDlPath;
  TDlLoadFlags dlLoadFlags= 0;
  bool threadedClose= false;
  TDynLib dlptr= nullptr;
};

}// namespace camp