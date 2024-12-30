// Copyright (c) Glyn Matthews 2012-2016.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef NETWORK_URI_DETAIL_TRANSLATE_INC
#define NETWORK_URI_DETAIL_TRANSLATE_INC

#include <string>
#include <algorithm>

namespace network {
namespace detail {
template <typename Source>
struct translate_impl;

template <>
struct translate_impl<std::string> {
  std::string operator()(const std::string &source) const { return source; }
};

template <int N>
struct translate_impl<char[N]> {
  std::string operator()(const char *source) const { return source; }
};

template <>
struct translate_impl<char *> {
  std::string operator()(const char *source) const { return source; }
};

template <>
struct translate_impl<const char *> {
  std::string operator()(const char *source) const { return source; }
};

template <int N>
struct translate_impl<const char[N]> {
  std::string operator()(const char *source) const { return source; }
};

template <>
struct translate_impl<std::wstring> {
  std::string operator()(const std::wstring &source) const {
    std::string ret(source.length(), 0);
    std::transform(
      source.begin(),
      source.end(),
      ret.begin(),
      [](wchar_t ch) { return static_cast<char>(ch); }
      );
    return ret;
  }
};

template <int N>
struct translate_impl<const wchar_t[N]> {
  std::string operator()(const wchar_t *source) const {
    translate_impl<std::wstring> impl;
    return impl(source);
  }
};

template <int N>
struct translate_impl<wchar_t[N]> {
  std::string operator()(const wchar_t *source) const {
    translate_impl<std::wstring> impl;
    return impl(source);
  }
};

template <>
struct translate_impl<wchar_t *> {
  std::string operator()(const wchar_t *source) const {
    translate_impl<std::wstring> impl;
    return impl(source);
  }
};

template <>
struct translate_impl<const wchar_t *> {
  std::string operator()(const wchar_t *source) const {
    translate_impl<std::wstring> impl;
    return impl(source);
  }
};

template <typename Source>
inline std::string translate(const Source &source) {
  translate_impl<Source> impl;
  return impl(source);
}

}  // namespace detail
}  // namespace network

#endif  // NETWORK_URI_DETAIL_TRANSLATE_INC
