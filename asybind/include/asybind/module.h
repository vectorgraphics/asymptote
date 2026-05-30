/*****
 * asybind/module.h
 * Author-facing C++ API for the Asymptote plugin SDK — Phase 0.
 *
 * Plugins include this header (typically via `#include <asybind/asybind.h>`)
 * and use the `ASY_MODULE(name, m)` macro plus `m.def("name", fn)` to expose
 * free functions to asy.
 *****/

#ifndef ASYBIND_MODULE_H
#define ASYBIND_MODULE_H

#include "abi.h"

#include <cstddef>
#include <cstring>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

namespace asy {

namespace detail {

/* Per-process current host API. Set when a module_handle is constructed
 * (i.e. on entry into the user's populate body). Allows casters whose
 * `spec()` needs to call into the host (e.g. callable<...> needing
 * make_function_type, result<T> needing result_class) to access the API
 * without threading it through every caster signature. */
inline const asybind_host_api_v1*& current_api() {
  static const asybind_host_api_v1* p = nullptr;
  return p;
}

/* Per-thunk-invocation current stack pointer. Set by the SDK's thunk
 * wrappers before calling the user body so that helpers such as
 * `callable<...>::operator()` and `ay::raise` can reach the host stack
 * without each user function having to thread it through. */
inline asybind_stack_ptr& current_stack() {
  static thread_local asybind_stack_ptr s = nullptr;
  return s;
}

/* RAII helper: scope the current stack to a thunk invocation. */
struct stack_scope {
  asybind_stack_ptr prev;
  stack_scope(asybind_stack_ptr s) : prev(current_stack()) { current_stack() = s; }
  ~stack_scope() { current_stack() = prev; }
};

/* === Per-type marshalling ============================================ */

template <class T> struct caster;

template <> struct caster<int> {
  static asybind_type_spec spec() { return { ASYBIND_INT, nullptr, nullptr }; }
  static int from_stack(asybind_stack_ptr s,
                        const asybind_host_api_v1* api) {
    return static_cast<int>(api->pop_int(s));
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, int v) {
    api->push_int(s, v);
  }
};

template <> struct caster<long long> {
  static asybind_type_spec spec() { return { ASYBIND_INT, nullptr, nullptr }; }
  static long long from_stack(asybind_stack_ptr s,
                              const asybind_host_api_v1* api) {
    return api->pop_int(s);
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, long long v) {
    api->push_int(s, v);
  }
};

template <> struct caster<double> {
  static asybind_type_spec spec() { return { ASYBIND_REAL, nullptr, nullptr }; }
  static double from_stack(asybind_stack_ptr s,
                           const asybind_host_api_v1* api) {
    return api->pop_real(s);
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, double v) {
    api->push_real(s, v);
  }
};

template <> struct caster<bool> {
  static asybind_type_spec spec() { return { ASYBIND_BOOL, nullptr, nullptr }; }
  static bool from_stack(asybind_stack_ptr s,
                         const asybind_host_api_v1* api) {
    return api->pop_bool(s) != 0;
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, bool v) {
    api->push_bool(s, v ? 1 : 0);
  }
};

template <> struct caster<std::string> {
  static asybind_type_spec spec() { return { ASYBIND_STRING, nullptr, nullptr }; }
  static std::string from_stack(asybind_stack_ptr s,
                                const asybind_host_api_v1* api) {
    const char* data = nullptr;
    std::size_t len = 0;
    api->pop_string(s, &data, &len);
    return std::string(data, len);
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api,
                       const std::string& v) {
    api->push_string(s, v.data(), v.size());
  }
};

/* === User-defined class types ======================================== *
 *
 * `caster<T*>` works for any C++ type T that has been registered with
 * `asy::class_<T>(m, "Name")`. The class_<T> constructor stashes the
 * host's class handle into the per-T `class_info<T>::handle()` slot;
 * thereafter caster<T*> consults that slot.
 */
template <class T>
struct class_info {
  static asybind_class_ptr& handle() {
    static asybind_class_ptr h = nullptr;
    return h;
  }
};

template <class T>
struct caster<T*> {
  static asybind_type_spec spec() {
    return { ASYBIND_USERPTR, class_info<T>::handle(), nullptr };
  }
  static T* from_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api) {
    return static_cast<T*>(api->pop_obj(s));
  }
  static void to_stack(asybind_stack_ptr s,
                       const asybind_host_api_v1* api, T* v) {
    api->push_obj(s, static_cast<void*>(v));
  }
};

/* === Function-traits ================================================== */

template <class F> struct fn_traits;

template <class R, class... A>
struct fn_traits<R(*)(A...)> {
  using result = R;
  using args   = std::tuple<A...>;
};

template <class C, class R, class... A>
struct fn_traits<R(C::*)(A...) const> {
  using result = R;
  using args   = std::tuple<A...>;
};

template <class C, class R, class... A>
struct fn_traits<R(C::*)(A...)> {
  using result = R;
  using args   = std::tuple<A...>;
};

template <class F>
struct fn_traits : fn_traits<decltype(&std::decay_t<F>::operator())> {};

/* === Argument spec arrays ============================================ */

template <class Tuple> struct spec_array;
template <class... A>
struct spec_array<std::tuple<A...>> {
  static constexpr int N = sizeof...(A);
  /* Build a fresh array on each call: spec() may depend on the runtime
   * value of class_info<T>::handle(), which is populated during plugin
   * setup. Plugins typically call this once per registration, so the
   * cost is negligible. */
  static const asybind_type_spec* values(asybind_type_spec* storage) {
    if constexpr (N == 0) {
      (void)storage;
      return nullptr;
    } else {
      asybind_type_spec specs[N] = {
        caster<std::decay_t<A>>::spec()...
      };
      for (std::size_t i = 0; i < N; ++i) storage[i] = specs[i];
      return storage;
    }
  }
};

/* === Call-from-stack dispatcher ====================================== */

/* Pops args (last argument first, matching asy's stack convention), then
 * invokes f and pushes the result. */
template <class F, class R, class... A, std::size_t... I>
void invoke_stub(F& f, asybind_stack_ptr s,
                 const asybind_host_api_v1* api,
                 std::tuple<A...>*, std::index_sequence<I...>) {
  constexpr std::size_t N = sizeof...(A);
  /* Pop into reverse-order placeholders so that arg N-1 is popped first. */
  std::tuple<std::optional<std::decay_t<A>>...> slots;
  (void)std::initializer_list<int>{
    (std::get<N - 1 - I>(slots).emplace(
        caster<std::decay_t<
            typename std::tuple_element<N - 1 - I,
                                        std::tuple<A...>>::type>>::from_stack(
            s, api)), 0)...
  };
  if constexpr (std::is_void_v<R>) {
    f(std::move(*std::get<I>(slots))...);
  } else {
    auto result = f(std::move(*std::get<I>(slots))...);
    caster<std::decay_t<R>>::to_stack(s, api, result);
  }
}

template <class F>
void invoke_zero(F& f, asybind_stack_ptr s,
                 const asybind_host_api_v1* api) {
  using R = typename fn_traits<F>::result;
  if constexpr (std::is_void_v<R>) {
    f();
  } else {
    auto result = f();
    caster<std::decay_t<R>>::to_stack(s, api, result);
  }
}

/* Static storage for a stateless callable so its thunk can be a plain
 * function pointer. We require the user callable to be convertible to a
 * function pointer; lambdas with captures are not supported in Phase 0. */
template <class F, class Tag>
struct fn_holder {
  static F& value() {
    static F instance{};
    return instance;
  }
};

/* Convert `caster<T>` to a type_spec, treating `void` as ASYBIND_VOID. */
template <class T, class = void>
struct caster_or_void {
  static asybind_type_spec spec() { return caster<T>::spec(); }
};
template <class T>
struct caster_or_void<T, std::enable_if_t<std::is_void_v<T>>> {
  static asybind_type_spec spec() { return { ASYBIND_VOID, nullptr, nullptr }; }
};

/* === module_ — author-facing module handle =========================== */

class module_handle {
public:
  module_handle(asybind_module_ptr m,
                const asybind_host_api_v1* api)
    : m_(m), api_(api) {
    detail::current_api() = api;
  }

  template <class F>
  module_handle& def(const char* name, F&& f) {
    using Decayed = std::decay_t<F>;
    using Traits = fn_traits<Decayed>;
    using Args   = typename Traits::args;
    using Res    = typename Traits::result;

    /* Store the callable in a per-(F,name) static slot so the C-linkage
     * thunk can find it without captures. */
    static Decayed slot{std::forward<F>(f)};

    asybind_thunk_t thunk = +[](asybind_stack_ptr s,
                                const asybind_host_api_v1* api) {
      detail::stack_scope ss(s);
      constexpr std::size_t N = std::tuple_size<Args>::value;
      if constexpr (N == 0) {
        invoke_zero(slot, s, api);
      } else {
        invoke_stub<Decayed, Res>(
            slot, s, api,
            static_cast<Args*>(nullptr),
            std::make_index_sequence<N>{});
      }
    };

    constexpr int N = static_cast<int>(std::tuple_size<Args>::value);
    asybind_type_spec restype = caster_or_void<Res>::spec();
    asybind_type_spec storage[(N > 0 ? N : 1)] = {};
    const asybind_type_spec* argspecs =
        spec_array<Args>::values(storage);
    api_->add_func(m_, name, thunk, restype, N, argspecs);
    return *this;
  }

  asybind_module_ptr           handle() const { return m_; }
  const asybind_host_api_v1*   api()    const { return api_; }

private:
  asybind_module_ptr           m_;
  const asybind_host_api_v1*   api_;
};

}  /* namespace detail */

using module_ = detail::module_handle;

[[noreturn]] inline void raise(const asybind_host_api_v1* api,
                               const char* msg) {
  api->raise(msg);
  std::abort();  /* unreachable; raise() does not return. */
}

/* Convenience overload using the current host API (set when the module's
 * populate body runs). */
[[noreturn]] inline void raise(const char* msg) {
  detail::current_api()->raise(msg);
  std::abort();
}
[[noreturn]] inline void raise(const std::string& msg) {
  raise(msg.c_str());
}

}  /* namespace asy */

/* === Plugin entry-point macro ======================================== */

#ifndef ASYBIND_ASY_VERSION_STRING
#  define ASYBIND_ASY_VERSION_STRING "unknown"
#endif

#define ASY_MODULE(NAME, MVAR)                                          \
  static void asybind_user_populate_##NAME(::asy::module_& MVAR);       \
  static void asybind_populate_##NAME(asybind_module_ptr __m,           \
                                      const asybind_host_api_v1* __api)\
  {                                                                     \
    ::asy::module_ MVAR(__m, __api);                                    \
    asybind_user_populate_##NAME(MVAR);                                 \
  }                                                                     \
  extern "C" ASY_EXPORT                                                 \
  const asybind_module_descriptor* asybind_init_v1(void)                \
  {                                                                     \
    static const asybind_module_descriptor desc = {                     \
      ASYBIND_ABI_VERSION,                                              \
      ASYBIND_ASY_VERSION_STRING,                                       \
      #NAME,                                                            \
      &asybind_populate_##NAME                                          \
    };                                                                  \
    return &desc;                                                       \
  }                                                                     \
  static void asybind_user_populate_##NAME(::asy::module_& MVAR)

#endif /* ASYBIND_MODULE_H */
