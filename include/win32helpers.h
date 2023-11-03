/**
 * @file win32helpers.h
 * @brief Windows-specific utility functions header
 * @author Supakorn "Jamie" Rassameemasmuang (jamievlin [at] outlook.com)
 */

#pragma once

#if defined(_WIN32)

#include <Windows.h>
#include "common.h"

namespace camp::w32
{

void checkResult(BOOL result, string const& message="");

class HandleRaiiWrapper
{
public:
    HandleRaiiWrapper(HANDLE const& handle);
    ~HandleRaiiWrapper();

    HandleRaiiWrapper(HandleRaiiWrapper const&) = delete;
    HandleRaiiWrapper& operator=(HandleRaiiWrapper const&) = delete;

    HandleRaiiWrapper(HandleRaiiWrapper&& other) noexcept;

    // already hold a handle, should not consume another one
    HandleRaiiWrapper& operator=(HandleRaiiWrapper&& other) = delete;

    HANDLE getHandle() const;

private:
    HANDLE hd;
};

} // namespace camp::w32

#endif
