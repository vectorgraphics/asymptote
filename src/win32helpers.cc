/**
 * @file win32helpers.cc
 * @brief Windows-specific utility functions
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 */

#if defined(_WIN32)
#include "win32helpers.h"
#include "errormsg.h"

using camp::reportError;

namespace camp::w32
{
void checkResult(BOOL result, string const& message)
{
  if (!result)
  {
    DWORD errorCode= GetLastError();
    ostringstream msg;
    msg << message << "; error code = 0x" << std::hex << errorCode << std::dec
        << "; Windows Message: " << getErrorMessage(errorCode);
    reportError(msg);
  }
}

void checkLStatus(LSTATUS result, string const& message)
{
  checkResult(result == ERROR_SUCCESS, message);
}

bool isProcessRunning(DWORD const& pid)
{
  if (pid == 0)
  {
    return true; // system idle is always running
  }

  HandleRaiiWrapper const processHandle(OpenProcess(PROCESS_QUERY_INFORMATION, false, pid));
  if (processHandle.getHandle() == nullptr)
  {
    // handle not in system, returns false
    return false;
  }

  DWORD exitCode=999;
  if (GetExitCodeProcess(processHandle.getHandle(), &exitCode))
  {
    if (exitCode == STILL_ACTIVE)
    {
      return true;
    }
    else
    {
      return false;
    }
  }

  return false;
}

#pragma region RegKeyWrapper

RegKeyWrapper::RegKeyWrapper(HKEY const& regKey)
    : key(regKey)
{
}
RegKeyWrapper::RegKeyWrapper() : key(nullptr)
{
}
RegKeyWrapper::~RegKeyWrapper()
{
  closeExistingKey();
}
RegKeyWrapper::RegKeyWrapper(RegKeyWrapper&& other) noexcept
        : key(std::exchange(other.key, nullptr))
{
}

RegKeyWrapper& RegKeyWrapper::operator=(RegKeyWrapper&& other) noexcept
{
  if (this != &other)
  {
    closeExistingKey();
    this->key = std::exchange(other.key, nullptr);
  }
  return *this;
}

HKEY RegKeyWrapper::getKey() const
{
  return key;
}

void RegKeyWrapper::closeExistingKey()
{
  if (this->key != nullptr)
  {
    RegCloseKey(this->key);
    this->key = nullptr;
  }
}

PHKEY RegKeyWrapper::put()
{
  closeExistingKey();
  return &(this->key);
}
void RegKeyWrapper::release()
{
  this->key = nullptr;
}

#pragma endregion

#pragma region HandleRaiiWrapper

HandleRaiiWrapper::HandleRaiiWrapper(HANDLE const& handle)
    : hd(handle)
{
}

HandleRaiiWrapper::~HandleRaiiWrapper()
{
  if (hd)
  {
    if (!CloseHandle(hd))
    {
      cerr << "Warning: Cannot close handle" << endl;
    }
  }
}

HandleRaiiWrapper::HandleRaiiWrapper(HandleRaiiWrapper&& other) noexcept
: hd(std::exchange(other.hd, nullptr))
{
}

HANDLE HandleRaiiWrapper::getHandle() const
{
  return hd;
}

LPHANDLE HandleRaiiWrapper::put()
{
  if (hd)
  {
    w32::checkResult(CloseHandle(hd));
    hd = nullptr;
  }

  return &hd;
}

#pragma endregion

string buildWindowsCmd(const mem::vector<string>& command)
{
  ostringstream out;
  for (auto it= command.begin(); it != command.end(); ++it)
  {
    out << '"' << *it << '"';
    if (std::next(it) != command.end())
    {
      out << ' ';
    }
  }

  return out.str();
}

string getErrorMessage(DWORD const& errorCode)
{
  LPSTR messageOut= nullptr;
  auto ret = FormatMessageA(
          FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
          nullptr,
          errorCode,
          MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
          reinterpret_cast<LPSTR>(&messageOut),
          0,
          nullptr
  );

  if (ret == 0)
  {
    return "Cannot determine error message";
  }

  string retString(messageOut);
  LocalFree(messageOut);
  
  return retString;
}

}// namespace camp::w32

#endif
