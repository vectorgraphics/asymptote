/*****
 * asybind/callable.h
 * Author-facing SDK type `asy::callable<R(A...)>` — Phase 2.
 *
 * A `callable<R(A...)>` is a thin, GC-traceable wrapper around an
 * asy `vm::callable*` (held opaquely as `asybind_callable_ptr`). It is
 * obtained either:
 *   - as a function argument (the SDK pops it from the stack via the
 *     `caster<callable<...>>::from_stack` specialization), or
 *   - by being stored as a member of a `class_<T>` instance (the host
 *     memory is GC-scanned, so the pointer stays live).
 *
 * Invocation pushes each argument through the matching `caster<>::to_stack`,
 * asks the host to dispatch the callable, then pops the result through
 * `caster<R>::from_stack`. Errors (via `asy::raise`) propagate as the
 * host's normal interrupted-execution unwind.
 *
 * Memory model: the SDK assumes that any C++ object holding a
 * `callable<...>` member is itself allocated in GC memory (via
 * `class_<T>::def(init<>())` or `asy::gc_new<T>()`); Boehm GC's
 * conservative scan then keeps the captured `vm::callable*` alive.
 * Holding a `callable<...>` on the C library stack for the lifetime of
 * a single thunk invocation is also safe — the asy stack frame above
 * pins the callable.
 *****/

#ifndef ASYBIND_CALLABLE_H
#define ASYBIND_CALLABLE_H

#include "module.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace asy {

template <class Sig> class callable;

template <class R, class... A>
class callable<R(A...)> {
public:
  callable() = default;

  explicit operator bool() const { return fn_ != nullptr; }

  /* Invoke the captured asy callable. Pushes args in source order
   * (so the last arg ends on top), dispatches, then pops the result
   * via caster<R>. */
  R operator()(A... args) const {
    const asybind_host_api_v1* api = detail::current_api();
    /* The host's stack lives on the asy side; we don't have a direct
     * pointer to it. For Phase 2 we rely on the invariant that this
     * call happens while a thunk is on top of the call stack (so the
     * thread's vm::stack* is accessible via the thunk's own argument).
     * The caller threads the stack pointer to us via a thread-local
     * set up at thunk entry by the invoke wrappers. */
    asybind_stack_ptr s = detail::current_stack();
    /* Push args in source order. */
    (void)std::initializer_list<int>{
      (detail::caster<std::decay_t<A>>::to_stack(s, api, args), 0)...
    };
    api->invoke_callable(s, fn_);
    if constexpr (std::is_void_v<R>) {
      (void)0;
    } else {
      return detail::caster<std::decay_t<R>>::from_stack(s, api);
    }
  }

  asybind_callable_ptr handle() const { return fn_; }

private:
  /* Type-caster constructs callables directly; users cannot make
   * non-null callables out of thin air. */
  template <class T> friend struct detail::caster;

  explicit callable(asybind_callable_ptr fn) : fn_(fn) {}

  asybind_callable_ptr fn_ = nullptr;
};

namespace detail {

/* === caster<callable<R(A...)>> ======================================= */

template <class R, class... A>
struct caster<callable<R(A...)>> {
  static asybind_type_spec spec() {
    const asybind_host_api_v1* api = current_api();
    asybind_type_spec restype = caster_or_void<R>::spec();
    constexpr int N = static_cast<int>(sizeof...(A));
    asybind_type_spec storage[(N > 0 ? N : 1)] = {};
    const asybind_type_spec* argspecs =
        spec_array<std::tuple<A...>>::values(storage);
    asybind_funty_ptr fnty =
        api->make_function_type(restype, N, argspecs);
    return { ASYBIND_FUNCTION, nullptr, fnty };
  }

  static callable<R(A...)> from_stack(asybind_stack_ptr s,
                                      const asybind_host_api_v1* api) {
    return callable<R(A...)>(api->pop_callable(s));
  }

  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api,
                       const callable<R(A...)>& v) {
    api->push_callable(s, v.handle());
  }
};

}  /* namespace detail */

}  /* namespace asy */

#endif /* ASYBIND_CALLABLE_H */
