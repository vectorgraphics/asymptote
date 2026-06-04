// Phase 3 test plugin: parameterized module + ay::Any.
//
// Exposes a single type parameter T and:
//   - Box: a tiny container holding an Any value, with set/get methods.
//   - id: identity for Any (round-trip through C++).
//   - apply: invoke an `Any(Any)` callable.
//
// Verified externally with T=int and T=string instantiations.
#include <asybind/asybind.h>

ASY_TEMPLATED_MODULE(phase3, m, "T") {
    auto T = m.type_param("T");
    (void)T;  // Phase 3 minimal: type_param's only role is documentation.

    struct Box {
        asy::Any value;
        void set(asy::Any v) { value = v; }
        asy::Any get() const { return value; }
    };

    asy::class_<Box>(m, "Box")
        .def(asy::init<>())
        .def<&Box::set>("set")
        .def<&Box::get>("get");

    m.def("id", [](asy::Any x) { return x; });

    m.def("apply", [](asy::callable<asy::Any(asy::Any)> f, asy::Any x) {
        return f(x);
    });
}
