/**
 * Search algorithms
 * @author Lobaskin Vasily
 * @data 31 March 2020
 * @copyright Boost Software License, Version 1.0
 */

#pragma once

#include <cstddef>

namespace network {
namespace algorithm {

template <typename IteratorBegT, typename IteratorEndT, typename RangeT>
IteratorBegT find_nth(IteratorBegT iteratorBeg, IteratorEndT iteratorEnd,
                      RangeT symbol, std::size_t pos) {
  static_assert(std::is_same<IteratorBegT, IteratorEndT>::value,
                "Iterator types are different");

  if (iteratorBeg > iteratorEnd) {
    std::swap(iteratorBeg, iteratorEnd);
  }

  std::size_t currentPos = -1;
  while (iteratorBeg != iteratorEnd) {
    if (*iteratorBeg == symbol) {
      ++currentPos;
      if (currentPos == pos) break;
    }
    ++iteratorBeg;
  }

  return iteratorBeg;
}

template <typename IteratorBegT, typename IteratorEndT, typename ConditionT>
bool all(IteratorBegT iteratorBeg, IteratorEndT iteratorEnd,
         ConditionT &&condition) {
  static_assert(std::is_same<IteratorBegT, IteratorEndT>::value,
                "Iterator types are different");

  if (iteratorBeg > iteratorEnd) {
    std::swap(iteratorBeg, iteratorEnd);
  }

  while (iteratorBeg != iteratorEnd) {
    if (!condition(*iteratorBeg)) return false;

    ++iteratorBeg;
  }

  return true;
}

template <typename ContainerT, typename RangeT>
typename ContainerT::iterator find_nth(ContainerT &str, RangeT symbol,
                                       std::size_t pos) {
  return algorithm::find_nth(str.begin(), str.end(), symbol, pos);
}

template <typename ContainerT, typename RangeT>
typename ContainerT::iterator find_last(ContainerT &str, RangeT symbol) {
  auto iter = algorithm::find_nth(str.rbegin(), str.rend(), symbol, 0);
  if (iter == str.rend()) {
    return str.end();
  }

  return (++iter).base();
}

template <typename ContainerT, typename ConditionT>
bool all(ContainerT const &container, ConditionT &&condition) {
  return all(container.cbegin(), container.cend(),
             std::forward<ConditionT>(condition));
}

}  // namespace algorithm
}  // namespace network