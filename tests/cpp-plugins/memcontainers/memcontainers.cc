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
    ay::mem::vector<long long> items;

    void   push(long long x)  { items.push_back(x); }
    long long pop()           {
        if (items.empty()) ay::raise("VecBag: empty");
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
    ay::mem::list<long long> items;

    void push_front(long long x) { items.push_front(x); }
    void push_back (long long x) { items.push_back(x); }
    long long front() const {
        if (items.empty()) ay::raise("ListBag: empty");
        return items.front();
    }
    long long back() const {
        if (items.empty()) ay::raise("ListBag: empty");
        return items.back();
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::deque -----------------------------------------------------
struct DequeBag {
    ay::mem::deque<long long> items;

    void   push_back (long long x) { items.push_back(x); }
    long long pop_front() {
        if (items.empty()) ay::raise("DequeBag: empty");
        long long x = items.front();
        items.pop_front();
        return x;
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::map<string, long long> -----------------------------------
struct StringMap {
    ay::mem::map<ay::mem::string, long long> items;

    void put(std::string k, long long v) {
        items[ay::mem::string(k.begin(), k.end())] = v;
    }
    long long get(std::string k) const {
        ay::mem::string key(k.begin(), k.end());
        auto it = items.find(key);
        if (it == items.end()) ay::raise("StringMap: missing key");
        return it->second;
    }
    bool contains(std::string k) const {
        ay::mem::string key(k.begin(), k.end());
        return items.find(key) != items.end();
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::unordered_map<long long, long long> ----------------------
struct IntMap {
    ay::mem::unordered_map<long long, long long> items;

    void put(long long k, long long v) { items[k] = v; }
    long long get(long long k) const {
        auto it = items.find(k);
        if (it == items.end()) ay::raise("IntMap: missing key");
        return it->second;
    }
    long long size() const { return static_cast<long long>(items.size()); }
};

// --- ay::mem::string ----------------------------------------------------
// Reverse a string using ay::mem::string so the working buffer comes
// from GC memory.
std::string reverse_via_mem_string(std::string s) {
    ay::mem::string buf(s.begin(), s.end());
    std::reverse(buf.begin(), buf.end());
    return std::string(buf.begin(), buf.end());
}

}  // namespace

ASY_MODULE(memcontainers, m) {
    ay::class_<VecBag>(m, "VecBag")
        .def(ay::init<>())
        .def<&VecBag::push>("push")
        .def<&VecBag::pop> ("pop")
        .def<&VecBag::sum> ("sum")
        .def<&VecBag::size>("size");

    ay::class_<ListBag>(m, "ListBag")
        .def(ay::init<>())
        .def<&ListBag::push_front>("push_front")
        .def<&ListBag::push_back> ("push_back")
        .def<&ListBag::front>     ("front")
        .def<&ListBag::back>      ("back")
        .def<&ListBag::size>      ("size");

    ay::class_<DequeBag>(m, "DequeBag")
        .def(ay::init<>())
        .def<&DequeBag::push_back> ("push_back")
        .def<&DequeBag::pop_front> ("pop_front")
        .def<&DequeBag::size>      ("size");

    ay::class_<StringMap>(m, "StringMap")
        .def(ay::init<>())
        .def<&StringMap::put>     ("put")
        .def<&StringMap::get>     ("get")
        .def<&StringMap::contains>("contains")
        .def<&StringMap::size>    ("size");

    ay::class_<IntMap>(m, "IntMap")
        .def(ay::init<>())
        .def<&IntMap::put>("put")
        .def<&IntMap::get>("get")
        .def<&IntMap::size>("size");

    m.def("reverse_via_mem_string", reverse_via_mem_string);
}
