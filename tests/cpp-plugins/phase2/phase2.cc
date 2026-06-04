// Phase 2 test plugin: exercises ay::callable<R(A...)> and ay::result<T>.
#include <asybind/asybind.h>
#include <string>

ASY_MODULE(phase2, m) {
    // ---- callable<R(A...)> ---------------------------------------
    // Apply an asy callback `int(int)` to an int and return the result.
    m.def("apply_int", [](asy::callable<int(int)> f, int x) {
        return f(x);
    });

    // Pass a string through a `string(string)` callback.
    m.def("apply_str", [](asy::callable<std::string(std::string)> f,
                          std::string s) {
        return f(s);
    });

    // Sum a callback applied to 0..n-1.
    m.def("sum_apply", [](asy::callable<int(int)> f, int n) {
        int total = 0;
        for (int i = 0; i < n; ++i) total += f(i);
        return total;
    });

    // ---- result<T> ----------------------------------------------
    // Safe-divide returns (found, value): not found when b == 0.
    m.def("safe_divide", [](int a, int b) {
        if (b == 0) return asy::result<int>(false, 0);
        return asy::result<int>(true, a / b);
    });

    // String lookup: returns (found, value) for a fixed table.
    m.def("lookup_name", [](int id) {
        switch (id) {
            case 1: return asy::result<std::string>(true, "alice");
            case 2: return asy::result<std::string>(true, "bob");
            default: return asy::result<std::string>(false, "");
        }
    });

    // ---- callable + result combined -----------------------------
    // Find first int in [0, n) where pred(i) is true.
    m.def("find_first", [](asy::callable<bool(int)> pred, int n) {
        for (int i = 0; i < n; ++i) {
            if (pred(i)) return asy::result<int>(true, i);
        }
        return asy::result<int>(false, 0);
    });
}
