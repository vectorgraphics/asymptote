// Plugin-author demonstration: GC-friendly STL containers via ay::mem.
//
// Each plugin object below stores its mutable state in a different
// `ay::mem::` container.  Because the objects themselves are created
// with `ay::gc_new` and the containers' backing buffers come from the
// host's scanned GC heap, every element — including the boxed `ay::Any`
// items returned to asy — stays reachable across collector cycles.
//
// No bespoke "gc array" helpers, no manual `alloc_obj` calls.

#include <asybind/asybind.h>

#include <algorithm>
#include <string>

namespace {

// --- ay::mem::vector ----------------------------------------------------
struct VecBag {
    asy::mem::vector<long long> items;

    void   push(long long x)  { items.push_back(x); }
    long long pop()           {
        if (items.empty()) asy::raise("VecBag: empty");
        long long x = items.back();
        items.pop_back();
        return x;
    }
    long long sum() const {
        long long s = 0;
        for (long long x : items) s += x;
        return s;
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::list ------------------------------------------------------
struct ListBag {
    asy::mem::list<long long> items;

    void push_front(long long x) { items.push_front(x); }
    void push_back (long long x) { items.push_back(x); }
    long long front() const {
        if (items.empty()) asy::raise("ListBag: empty");
        return items.front();
    }
    long long back() const {
        if (items.empty()) asy::raise("ListBag: empty");
        return items.back();
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::deque -----------------------------------------------------
struct DequeBag {
    asy::mem::deque<long long> items;

    void   push_back (long long x) { items.push_back(x); }
    long long pop_front() {
        if (items.empty()) asy::raise("DequeBag: empty");
        long long x = items.front();
        items.pop_front();
        return x;
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::map<string, long long> -----------------------------------
struct StringMap {
    asy::mem::map<asy::mem::string, long long> items;

    void put(std::string k, long long v) {
        items[asy::mem::string(k.begin(), k.end())] = v;
    }
    long long get(std::string k) const {
        asy::mem::string key(k.begin(), k.end());
        auto it = items.find(key);
        if (it == items.end()) asy::raise("StringMap: missing key");
        return it->second;
    }
    bool contains(std::string k) const {
        asy::mem::string key(k.begin(), k.end());
        return items.find(key) != items.end();
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::unordered_map<long long, long long> ----------------------
struct IntMap {
    asy::mem::unordered_map<long long, long long> items;

    void put(long long k, long long v) { items[k] = v; }
    long long get(long long k) const {
        auto it = items.find(k);
        if (it == items.end()) asy::raise("IntMap: missing key");
        return it->second;
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::string ----------------------------------------------------
// Reverse a string using ay::mem::string so the working buffer comes
// from GC memory.
std::string reverse_via_mem_string(std::string s) {
    asy::mem::string buf(s.begin(), s.end());
    std::reverse(buf.begin(), buf.end());
    return std::string(buf.begin(), buf.end());
}

}  // namespace

ASY_MODULE(memcontainers, m) {
    asy::class_<VecBag>(m, "VecBag")
        .def(asy::init<>())
        .def<&VecBag::push>("push")
        .def<&VecBag::pop> ("pop")
        .def<&VecBag::sum> ("sum")
        .def<&VecBag::size>("size");

    asy::class_<ListBag>(m, "ListBag")
        .def(asy::init<>())
        .def<&ListBag::push_front>("push_front")
        .def<&ListBag::push_back> ("push_back")
        .def<&ListBag::front>     ("front")
        .def<&ListBag::back>      ("back")
        .def<&ListBag::size>      ("size");

    asy::class_<DequeBag>(m, "DequeBag")
        .def(asy::init<>())
        .def<&DequeBag::push_back> ("push_back")
        .def<&DequeBag::pop_front> ("pop_front")
        .def<&DequeBag::size>      ("size");

    asy::class_<StringMap>(m, "StringMap")
        .def(asy::init<>())
        .def<&StringMap::put>     ("put")
        .def<&StringMap::get>     ("get")
        .def<&StringMap::contains>("contains")
        .def<&StringMap::size>    ("size");

    asy::class_<IntMap>(m, "IntMap")
        .def(asy::init<>())
        .def<&IntMap::put>("put")
        .def<&IntMap::get>("get")
        .def<&IntMap::size>("size");

    m.def("reverse_via_mem_string", reverse_via_mem_string);
}
