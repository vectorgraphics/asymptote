// Copyright 2016 Glyn Matthews.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef NETWORK_DETAIL_URI_PARTS_INC
#define NETWORK_DETAIL_URI_PARTS_INC

#include <string>
#include <utility>
#include <iterator>
#include <network/optional.hpp>
#include <network/string_view.hpp>

namespace network {
namespace detail {
class uri_part {
 public:
  typedef string_view::value_type value_type;
  typedef string_view::iterator iterator;
  typedef string_view::const_iterator const_iterator;
  typedef string_view::const_pointer const_pointer;
  typedef string_view::size_type size_type;
  typedef string_view::difference_type difference_type;

  uri_part() noexcept = default;

  uri_part(const_iterator first, const_iterator last) noexcept
      : first(first), last(last) {}

  const_iterator begin() const noexcept { return first; }

  const_iterator end() const noexcept { return last; }

  bool empty() const noexcept { return first == last; }

  std::string to_string() const { return std::string(first, last); }

  const_pointer ptr() const noexcept {
    assert(first != last);
    return first;
  }

  difference_type length() const noexcept { return last - first; }

  string_view to_string_view() const noexcept {
    return string_view(ptr(), length());
  }

 private:
  const_iterator first, last;
};

struct hierarchical_part {
  hierarchical_part() = default;

  optional<uri_part> user_info;
  optional<uri_part> host;
  optional<uri_part> port;
  optional<uri_part> path;

  void clear() {
    user_info = nullopt;
    host = nullopt;
    port = nullopt;
    path = nullopt;
  }
};

struct uri_parts {
  uri_parts() = default;

  optional<uri_part> scheme;
  hierarchical_part hier_part;
  optional<uri_part> query;
  optional<uri_part> fragment;

  void clear() {
    scheme = nullopt;
    hier_part.clear();
    query = nullopt;
    fragment = nullopt;
  }
};
}  // namespace detail
}  // namespace network

#endif  // NETWORK_DETAIL_URI_PARTS_INC
