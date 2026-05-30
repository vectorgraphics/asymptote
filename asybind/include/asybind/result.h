/*****
 * asybind/result.h
 * Author-facing SDK type `asy::result<T>` — Phase 2.
 *
 * `result<T>` is a thin (found?, value) pair. When returned from a
 * bltin, the SDK pushes the value (only if `found` is true) and asks
 * the host to allocate a synthesized `result_T` instance whose two
 * readonly fields the asy wrapper destructures:
 *
 *     var r = core.lookup(item);
 *     if (r.found) ... r.value ...
 *
 * The host caches the `result_T` record per element-type spec so that
 * repeated registrations of `result<int>` (say) refer to the same
 * asy-side type.
 *****/

#ifndef ASYBIND_RESULT_H
#define ASYBIND_RESULT_H

#include "module.h"

#include <type_traits>

namespace asy {

template <class T>
struct result {
  bool found;
  T    value;

  result() : found(false), value() {}
  result(bool f, T v) : found(f), value(std::move(v)) {}
};

namespace detail {

template <class T>
struct caster<result<T>> {
  /* No SDK-side cache: across multiple templated-module instantiations
   * the same `result<T>` C++ type can correspond to distinct asy
   * record types (e.g. `result<Any>` where the underlying T resolves
   * differently per instantiation). The host already deduplicates
   * `result_class` calls keyed on the element spec, so the per-call
   * cost is one hash lookup. */
  static asybind_class_ptr handle() {
    const asybind_host_api_v1* api = current_api();
    asybind_type_spec elem = caster<std::decay_t<T>>::spec();
    return api->result_class(elem);
  }

  static asybind_type_spec spec() {
    return { ASYBIND_USERPTR, handle(), nullptr };
  }

  /* Result values are produced on the asy side (as record instances)
   * but ordinarily not consumed by C++; `from_stack` is therefore not
   * provided. Returning a result<T> from a bltin is the primary use. */
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api,
                       const result<T>& v) {
    asybind_class_ptr cls = handle();
    if (v.found) {
      /* Push the value first; push_result pops it off the top into the
       * synthesized record instance. */
      caster<std::decay_t<T>>::to_stack(s, api, v.value);
      api->push_result(s, cls, 1);
    } else {
      api->push_result(s, cls, 0);
    }
  }
};

}  /* namespace detail */

}  /* namespace asy */

#endif /* ASYBIND_RESULT_H */
