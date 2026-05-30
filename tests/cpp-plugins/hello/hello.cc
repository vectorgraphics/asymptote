// Phase 0 fixture: a trivial C++ plugin exposing a single function.
#include <asybind/asybind.h>
#include <string>

ASY_MODULE(hello, m) {
    m.def("hello", [] { return std::string("hi"); });
    m.def("greet", [](std::string name) { return std::string("hello, ") + name; });
    m.def("sum", [](int a, int b) { return a + b; });
}
