#pragma once
#if __cplusplus >= 201703L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201703L)
#include <filesystem>
namespace filesystem = std::filesystem;

#else

#include <ghc/filesystem.hpp>
namespace filesystem = ghc::filesystem;

#endif
