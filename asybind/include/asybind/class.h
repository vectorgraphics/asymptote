/*****
 * asybind/class.h
 * Author-facing C++ class registration for the Asymptote plugin SDK
 * (Phase 1).
 *
 * Usage:
 *   ASY_MODULE(mymod, m) {
 *     struct Box { int value = 42; int size() const { return value; } };
 *     asy::class_<Box>(m, "Box")
 *         .def(asy::init<>())
 *         .def<&Box::size>("size")
 *         .def_readonly<&Box::value>("value");
 *   }
 *
 * The `def<&Box::method>("name")` and `def_readonly<&Box::field>("name")`
 * forms use C++17 non-type template parameters so that every registration
 * gets its own statically-generated thunk without relying on per-method
 * static slots that could collide between methods of the same signature.
 *
 * The constructor function is registered as a free function named after
 * the class (e.g. `Box()`) in the module's env, returning a `Box*`. The
 * recommended asy syntax is therefore `Box b = Box();`.
 *****/

#ifndef ASYBIND_CLASS_H
#define ASYBIND_CLASS_H

#include "module.h"

#include <new>
#include <type_traits>

namespace asy {

/* Tag type used to select the constructor overload of class_<T>::def. */
template <class... A>
struct init {};

namespace detail {

/* === Member-pointer traits =========================================== */

template <class PMFn> struct mfn_traits;

template <class C, class R, class... A>
struct mfn_traits<R (C::*)(A...)> {
  using class_type = C;
  using result     = R;
  using args       = std::tuple<A...>;
  static constexpr bool is_const = false;
};

template <class C, class R, class... A>
struct mfn_traits<R (C::*)(A...) const> {
  using class_type = C;
  using result     = R;
  using args       = std::tuple<A...>;
  static constexpr bool is_const = true;
};

template <class PMD> struct pmd_traits;

template <class C, class R>
struct pmd_traits<R C::*> {
  using class_type = C;
  using field_type = R;
};

/* === Method thunk ==================================================== */

/* Helper: pop `N` args (reverse), then invoke (self->*MFn)(args...). */
template <auto MFn, class C, class R, class... A, std::size_t... I>
void invoke_method(C* self, asybind_stack_ptr s,
                   const asybind_host_api_v1* api,
                   std::tuple<A...>*, std::index_sequence<I...>) {
  constexpr std::size_t N = sizeof...(A);
  std::tuple<std::optional<std::decay_t<A>>...> slots;
  (void)std::initializer_list<int>{
    (std::get<N - 1 - I>(slots).emplace(
        caster<std::decay_t<
            typename std::tuple_element<N - 1 - I,
                                        std::tuple<A...>>::type>>::from_stack(
            s, api)), 0)...
  };
  if constexpr (std::is_void_v<R>) {
    (self->*MFn)(std::move(*std::get<I>(slots))...);
  } else {
    auto r = (self->*MFn)(std::move(*std::get<I>(slots))...);
    caster<std::decay_t<R>>::to_stack(s, api, r);
  }
}

/* Thunk: pops receiver (USERPTR), then args, then calls the method. */
template <auto MFn>
void method_thunk(asybind_stack_ptr s,
                  const asybind_host_api_v1* api) {
  stack_scope ss(s);
  using Traits = mfn_traits<decltype(MFn)>;
  using C      = typename Traits::class_type;
  using R      = typename Traits::result;
  using Args   = typename Traits::args;

  C* self = static_cast<C*>(api->pop_obj(s));
  constexpr std::size_t N = std::tuple_size<Args>::value;
  if constexpr (N == 0) {
    if constexpr (std::is_void_v<R>) {
      (self->*MFn)();
    } else {
      auto r = (self->*MFn)();
      caster<std::decay_t<R>>::to_stack(s, api, r);
    }
  } else {
    invoke_method<MFn, C, R>(self, s, api,
                             static_cast<Args*>(nullptr),
                             std::make_index_sequence<N>{});
  }
}

/* Thunk: pops receiver (USERPTR), then pushes self->*PMD. */
template <auto PMD>
void readonly_field_thunk(asybind_stack_ptr s,
                          const asybind_host_api_v1* api) {
  stack_scope ss(s);
  using Traits = pmd_traits<decltype(PMD)>;
  using C      = typename Traits::class_type;
  using F      = typename Traits::field_type;
  C* self = static_cast<C*>(api->pop_obj(s));
  caster<std::decay_t<F>>::to_stack(s, api, self->*PMD);
}

}  /* namespace detail */

/* === class_<T> ======================================================= */

template <class T>
class class_ {
public:
  class_(module_& m, const char* name)
    : m_(m), name_(name),
      handle_(m.api()->create_class(m.handle(), name)) {
    detail::class_info<T>::handle() = handle_;
  }

  /* Constructor registration. */
  class_& def(init<>) {
    asybind_thunk_t thunk = +[](asybind_stack_ptr s,
                                const asybind_host_api_v1* api) {
      detail::stack_scope ss(s);
      void* mem = api->alloc_obj(sizeof(T));
      ::new (mem) T();
      api->push_obj(s, mem);
    };
    asybind_type_spec restype{ ASYBIND_USERPTR, handle_, nullptr };
    m_.api()->add_func(m_.handle(), name_, thunk, restype,
                       /*nargs=*/0, /*argtypes=*/nullptr);
    return *this;
  }

  /* Method registration: usage `cls.def<&T::method>("name")`. */
  template <auto PMFn>
  class_& def(const char* mname) {
    using Traits = detail::mfn_traits<decltype(PMFn)>;
    static_assert(std::is_same_v<typename Traits::class_type, T>,
                  "asybind: def<&T::m>(name) — method must belong to T");
    using R    = typename Traits::result;
    using Args = typename Traits::args;
    constexpr int N = static_cast<int>(std::tuple_size<Args>::value);

    asybind_thunk_t thunk = &detail::method_thunk<PMFn>;
    asybind_type_spec restype =
        detail::caster_or_void<R>::spec();
    asybind_type_spec storage[(N > 0 ? N : 1)] = {};
    const asybind_type_spec* argspecs =
        detail::spec_array<Args>::values(storage);
    m_.api()->add_method(handle_, mname, thunk, restype, N, argspecs);
    return *this;
  }

  /* Readonly field: usage `cls.def_readonly<&T::field>("name")`. */
  template <auto PMD>
  class_& def_readonly(const char* fname) {
    using Traits = detail::pmd_traits<decltype(PMD)>;
    static_assert(std::is_same_v<typename Traits::class_type, T>,
                  "asybind: def_readonly<&T::f>(name) — field must belong to T");
    using F = typename Traits::field_type;
    asybind_thunk_t thunk = &detail::readonly_field_thunk<PMD>;
    asybind_type_spec type = detail::caster<std::decay_t<F>>::spec();
    m_.api()->add_readonly_field(handle_, fname, thunk, type);
    return *this;
  }

private:
  module_&             m_;
  const char*          name_;
  asybind_class_ptr    handle_;
};

}  /* namespace asy */

#endif /* ASYBIND_CLASS_H */
