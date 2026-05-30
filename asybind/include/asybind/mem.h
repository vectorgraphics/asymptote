/*****
 * asybind/mem.h
 * GC-friendly STL allocator and container aliases for plugin authors.
 *
 * Plugins that store data which is reachable across asy boundaries (or
 * across any GC pause) must place that data in memory the Boehm GC can
 * scan.  The host API exposes `alloc_obj(n)` for this purpose, but
 * dropping that into every container's storage path is awkward.  This
 * header packages it as a standard C++ Allocator and provides the
 * usual STL container aliases parameterised by it.  In particular:
 *
 *     ay::mem::vector<T>
 *     ay::mem::list<T>
 *     ay::mem::deque<T>
 *     ay::mem::string
 *     ay::mem::map<K, V>
 *     ay::mem::set<K>
 *     ay::mem::unordered_map<K, V>
 *     ay::mem::unordered_set<K>
 *
 * are very nearly drop-in replacements for their std:: counterparts.
 *
 * Usage notes:
 *   * Containers placed as members of objects allocated via
 *     `ay::gc_new<T>` are fully GC-tracked: the enclosing object lives
 *     in GC memory, its allocator-backed buffers live in GC memory, and
 *     the conservative scan finds every pointer kept inside either.
 *   * The allocator is stateless and always-equal, so containers can
 *     be moved, swapped, and copied without surprises.
 *   * `deallocate` is a no-op; the GC reclaims storage automatically.
 *   * Allocation failure (which should be vanishingly rare in
 *     practice) raises an asy error rather than throwing, matching the
 *     rest of the asybind SDK.
 *
 * This header intentionally keeps its dependencies to the C++ standard
 * library plus asybind/module.h (for `current_api()` and `ay::raise`).
 *****/

#ifndef ASYBIND_MEM_H
#define ASYBIND_MEM_H

#include "module.h"

#include <cstddef>
#include <deque>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace asy {
namespace mem {

/* -------------------------------------------------------------------
 * gc_allocator<T>
 *
 * A stateless C++17 Allocator that obtains storage from the asy host's
 * scanned GC heap via the ABI's `alloc_obj` thunk.
 * ------------------------------------------------------------------- */
template <class T>
struct gc_allocator {
  using value_type                             = T;
  using size_type                              = std::size_t;
  using difference_type                        = std::ptrdiff_t;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap            = std::true_type;
  using is_always_equal                        = std::true_type;

  template <class U> struct rebind { using other = gc_allocator<U>; };

  gc_allocator() noexcept = default;
  gc_allocator(const gc_allocator&) noexcept = default;
  template <class U>
  gc_allocator(const gc_allocator<U>&) noexcept {}

  T* allocate(std::size_t n) {
    if (n == 0) return nullptr;
    void* p = ::asy::detail::current_api()->alloc_obj(n * sizeof(T));
    if (p == nullptr) ::asy::raise("asy::mem::gc_allocator: allocation failed");
    return static_cast<T*>(p);
  }

  void deallocate(T* /*p*/, std::size_t /*n*/) noexcept {
    /* GC reclaims storage; nothing to do. */
  }
};

template <class T, class U>
inline bool operator==(const gc_allocator<T>&, const gc_allocator<U>&) noexcept {
  return true;
}
template <class T, class U>
inline bool operator!=(const gc_allocator<T>&, const gc_allocator<U>&) noexcept {
  return false;
}

/* -------------------------------------------------------------------
 * Container aliases.
 *
 * Each alias mirrors the corresponding std:: container with the
 * default allocator replaced by `gc_allocator`.  Default predicates
 * (std::less, std::hash, std::equal_to) are kept identical to the
 * std:: versions so user code can drop the `std::` prefix in most
 * places without further changes.
 * ------------------------------------------------------------------- */

template <class T>
using vector = std::vector<T, gc_allocator<T>>;

template <class T>
using list = std::list<T, gc_allocator<T>>;

template <class T>
using deque = std::deque<T, gc_allocator<T>>;

using string = std::basic_string<char, std::char_traits<char>,
                                 gc_allocator<char>>;

template <class K, class V, class Cmp = std::less<K>>
using map = std::map<K, V, Cmp,
                     gc_allocator<std::pair<const K, V>>>;

template <class K, class V, class Cmp = std::less<K>>
using multimap = std::multimap<K, V, Cmp,
                               gc_allocator<std::pair<const K, V>>>;

template <class K, class Cmp = std::less<K>>
using set = std::set<K, Cmp, gc_allocator<K>>;

template <class K, class Cmp = std::less<K>>
using multiset = std::multiset<K, Cmp, gc_allocator<K>>;

template <class K, class V,
          class H  = std::hash<K>,
          class Eq = std::equal_to<K>>
using unordered_map =
    std::unordered_map<K, V, H, Eq,
                       gc_allocator<std::pair<const K, V>>>;

template <class K, class V,
          class H  = std::hash<K>,
          class Eq = std::equal_to<K>>
using unordered_multimap =
    std::unordered_multimap<K, V, H, Eq,
                            gc_allocator<std::pair<const K, V>>>;

template <class K,
          class H  = std::hash<K>,
          class Eq = std::equal_to<K>>
using unordered_set = std::unordered_set<K, H, Eq, gc_allocator<K>>;

template <class K,
          class H  = std::hash<K>,
          class Eq = std::equal_to<K>>
using unordered_multiset =
    std::unordered_multiset<K, H, Eq, gc_allocator<K>>;

}  // namespace mem
}  // namespace asy

#endif  // ASYBIND_MEM_H
