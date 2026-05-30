// Phase 3 test plugin: parameterized module + ay::Any.
//
// Exposes a single type parameter T and:
//   - Box: a tiny container holding an Any value, with set/get methods.
//   - id: identity for Any (round-trip through C++).
//   - apply: invoke an `Any(Any)` callable.
//
// Verified externally with T=int and T=string instantiations.
#include <asybind/asybind.h>

namespace ay = asy;

ASY_TEMPLATED_MODULE(phase3, m, "T") {
    auto T = m.type_param("T");
    (void)T;  // Phase 3 minimal: type_param's only role is documentation.

    struct Box {
        ay::Any value;
        void set(ay::Any v) { value = v; }
        ay::Any get() const { return value; }
    };

    ay::class_<Box>(m, "Box")
        .def(ay::init<>())
        .def<&Box::set>("set")
        .def<&Box::get>("get");

    m.def("id", [](ay::Any x) { return x; });

    m.def("apply", [](ay::callable<ay::Any(ay::Any)> f, ay::Any x) {
        return f(x);
    });
}
