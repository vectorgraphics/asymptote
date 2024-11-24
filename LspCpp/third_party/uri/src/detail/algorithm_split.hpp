/**
 * Search algorithms
 * @author Lobaskin Vasily
 * @data 31 March 2020
 * @copyright Boost Software License, Version 1.0
 */

#include <type_traits>

namespace network {
namespace algorithm {

template <typename ContainerT, class SequenceT, typename SplitterT,
          typename std::enable_if<std::is_fundamental<SplitterT>::value>::type
              * = nullptr>
bool split(ContainerT &container, SequenceT const &str, SplitterT symbol) {
  using PartT = typename ContainerT::value_type;
  static_assert(std::is_same<typename SequenceT::value_type, SplitterT>::value,
                "Splitter type doesn't match sequence inner type");

  std::size_t sequenceStart = 0;
  for (std::size_t i = 0, len = str.size(); i <= len; ++i) {
    if (str[i] != symbol && i != len) continue;

    std::size_t substrLen = i - sequenceStart;
    if (substrLen > 0) {
      PartT part{str.cbegin() + sequenceStart, str.cbegin() + i};
      container.emplace_back(std::move(part));
    } else {
      container.emplace_back(PartT{});
    }
    sequenceStart = i + 1;
  }

  return true;
}

template <typename ContainerT, class SequenceT, typename SplitterT,
          typename std::enable_if<!std::is_fundamental<SplitterT>::value>::type
              * = nullptr>
bool split(ContainerT &container, SequenceT const &str, SplitterT splitter) {
  using PartT = typename ContainerT::value_type;
  static_assert(
      std::is_same<typename ContainerT::value_type, std::string>::value,
      "Invalid container type, only string is supported");

  bool isEqual = false;
  std::size_t sequenceLen = splitter.size();
  std::size_t sequenceStart = 0;
  for (std::size_t i = 0, len = str.size(); i <= len; ++i) {
    isEqual = true;
    for (std::size_t j = 0; j < sequenceLen; ++j) {
      if (str[i + j] != splitter[j]) {
        isEqual = false;
        break;
      }
    }
    if (!isEqual && i != len) continue;

    std::size_t substrLen = i - sequenceStart;
    if (substrLen > 0) {
      PartT part{str.cbegin() + sequenceStart, str.cbegin() + i};
      container.emplace_back(std::move(part));
    } else {
      container.emplace_back(PartT{});
    }
    sequenceStart = i + sequenceLen;
    i += sequenceLen - 1;
  }

  return true;
}

}  // namespace algorithm
}  // namespace network
