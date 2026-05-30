/*****
 * asybind/any.h
 * Author-facing SDK type `asy::Any` — Phase 3.
 *
 * An `Any` is an opaque wrapper around an asy value of (asy-side) type
 * `T`, where `T` is the *first* type parameter of the enclosing
 * parameterized module. The C++ side never inspects the contents — it
 * only stores Any values, passes them through callables, and returns
 * them to asy. All conversion happens at the asy/C++ boundary.
 *
 * Memory model: the underlying storage is a host-allocated `vm::item`
 * (GC-tracked). `Any` itself is a thin pointer wrapper that is safe to
 * hold transiently on the C library stack, store inside a class_<T>
 * struct field (the host's GC scan keeps the pointed-to item alive),
 * or pass through `callable<>` invocations.
 *
 * Limitations (Phase 3 minimal):
 *   - Only a single type parameter T is supported; the type spec for
 *     `Any` resolves to `get_resolved_type(module, 0)`. Multi-parameter
 *     modules want an `Any<I>` template tagged by parameter index;
 *     deferred to a later phase.
 *   - No equality, no hash, no `cast<T>()`. The wrapper has all the
 *     domain knowledge it needs; the core stays oblivious.
 *****/

#ifndef ASYBIND_ANY_H
#define ASYBIND_ANY_H

#include "module.h"

namespace asy {

class Any {
public:
  Any() = default;

  bool is_null() const { return h_ == nullptr; }
  asybind_any_ptr handle() const { return h_; }

private:
  template <class T> friend struct detail::caster;
  explicit Any(asybind_any_ptr h) : h_(h) {}

  asybind_any_ptr h_ = nullptr;
};

namespace detail {

template <>
struct caster<Any> {
  static asybind_type_spec spec() {
    const asybind_host_api_v1* api = current_api();
    return api->get_resolved_type(current_module(), 0);
  }
  static Any from_stack(asybind_stack_ptr s,
                        const asybind_host_api_v1* api) {
    return Any(api->pop_any(s));
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, const Any& v) {
    api->push_any(s, v.handle());
  }
};

}  /* namespace detail */

}  /* namespace asy */

#endif /* ASYBIND_ANY_H */
