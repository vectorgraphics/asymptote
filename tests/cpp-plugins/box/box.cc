// Phase 1 fixture: a C++ class with one field and one method.
#include <asybind/asybind.h>

struct Box {
    int value = 42;
    int size() const { return value; }
};

ASY_MODULE(box, m) {
    asy::class_<Box>(m, "Box")
        .def(asy::init<>())
        .def<&Box::size>("size")
        .def_readonly<&Box::value>("value");
}
