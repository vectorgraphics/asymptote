#pragma once

#include <string>

#if defined(_MSC_VER)
#  if defined(_M_X64)
std::string const ARCH_STRING= "x64";
#  elif defined(_M_ARM)
std::string const ARCH_STRING= "ARM";
#  elif defined(_M_IX86)
std::string const ARCH_STRING= "x86";// is this macro even needed?
#  elif defined(_M_ARM64)
std::string const ARCH_STRING= "ARM64";
#  else
std::string const ARCH_STRING= "Unknown architecture";
#  endif
std::string const ASY_COMPILER_INFO=
        "MSVC v" + std::to_string(_MSC_VER) + " (" + ARCH_STRING + ")";
#elif defined(__GNUC__)// gcc-class compilers

#  if defined(__x86_64__)
std::string const ARCH_STRING= "x64";
#  elif defined(__arm__)
std::string const ARCH_STRING= "ARM";
#  elif defined(__aarch64__)
std::string const ARCH_STRING= "ARM64";
#  else
std::string const ARCH_STRING= "Unknown architecture";
#  endif

#  if defined(__clang__)// for clang
std::string const ASY_COMPILER_INFO=
        "clang " + __clang_version__ + " (" + ARCH_STRING + ")";
#  else                 // for gcc
std::string const ASY_COMPILER_INFO= "GCC " + std::to_string(__GNUC__) + "." +
                                     std::to_string(__GNUC_MINOR__) + "." +
                                     std::to_string(__GNUC_PATCHLEVEL__) +
                                     " (" + ARCH_STRING + ")";
#  endif


#else
const std::string ASY_COMPILER_INFO= "unknown compiler"
#endif
