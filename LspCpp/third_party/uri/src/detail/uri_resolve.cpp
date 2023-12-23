// Copyright 2013-2016 Glyn Matthews.
// Copyright 2013 Hannes Kamecke.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include "uri_resolve.hpp"
#include "algorithm_find.hpp"

using namespace network::algorithm;
using network::uri;
using network::string_view;
namespace network_detail = network::detail;

namespace {

// remove_dot_segments
inline void remove_last_segment(std::string &path) {
  while (!path.empty()) {
    if (path.back() == '/') {
      path.pop_back();
      break;
    }
    path.pop_back();
  }
}

inline bool starts_with(std::string const &str, const char *range) {
  return str.find(range) == 0;
}

// implementation of http://tools.ietf.org/html/rfc3986#section-5.2.4
static std::string remove_dot_segments(std::string input) {
  std::string result;

  while (!input.empty()) {
    if (starts_with(input, "../")) {
      input.erase(0, 3);
    } else if (starts_with(input, "./")) {
      input.erase(0, 2);
    } else if (starts_with(input, "/./")) {
      input.erase(0, 2);
      input.front() = '/';
    } else if (input == "/.") {
      input.erase(0, 1);
      input.front() = '/';
    } else if (starts_with(input, "/../")) {
      input.erase(0, 3);
      remove_last_segment(result);
    } else if (starts_with(input, "/..")) {
      input.erase(0, 2);
      input.front() = '/';
      remove_last_segment(result);
    } else if (all(input, [](char ch) { return ch == '.'; })) {
      input.clear();
    } else {
      int n = (input.front() == '/') ? 1 : 0;
      std::string::iterator slash = find_nth(input, '/', n);
      result.append(std::begin(input), slash);
      input.erase(std::begin(input), slash);
    }
  }
  return result;
}

}  // namespace

std::string network_detail::remove_dot_segments(string_view path) {
  return ::remove_dot_segments(path.to_string());
}

// implementation of http://tools.ietf.org/html/rfc3986#section-5.2.3
std::string network_detail::merge_paths(const uri &base, const uri &reference) {
  std::string result;

  if (!base.has_path() || base.path().empty()) {
    result = "/";
  } else {
    const auto &base_path = base.path();
    auto last_slash = algorithm::find_last(base_path, '/');
    if (last_slash != base_path.cend()) ++last_slash;
    result.append(std::begin(base_path), last_slash);
  }
  if (reference.has_path()) {
    result.append(reference.path().to_string());
  }
  return remove_dot_segments(string_view(result));
}
