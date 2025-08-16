typedef import(T);

from collections.iter(T=T) access Iter_T, Iterable_T;
from mapArray(Src=T, Dst=int) access map;

struct Array_T {
  restricted T[] data;
  restricted int hash() = null;
  // Simulate the brackets and iteration of an array.
  restricted T operator [](int i) { return data[i]; }
  restricted void operator [=](int i, T x) { data[i] = x; }
  restricted Iter_T operator iter() { return Iter_T(data); }

  // Match the array's fields as closely as possible.
  restricted int length() { return data.length; }
  restricted void cyclic(bool b) { data.cyclic = b; }
  restricted bool cyclic() { return data.cyclic; }
  restricted int[] keys() { return data.keys; }
  restricted T push(T x);
  restricted void append(T[] x);
  restricted void append(Array_T x) { data.append(x.data); }
  restricted T pop();
  restricted void insert(int i ... T[] x);
  restricted void delete(int i, int j=i);
  restricted bool initialized(int n);

  void operator init(T[] data, int hashElement(T x) = null) {
    this.data = data;
    if (hashElement != null) this.hash = new int() {
      return hash(map(hashElement, this.data));
    };
    this.push = data.push;
    this.append = data.append;
    this.pop = data.pop;
    this.insert = data.insert;
    this.delete = data.delete;
    this.initialized = data.initialized;
  }

  // Non-vectorized operator== and operator!=.
  autounravel bool operator ==(Array_T a, Array_T b) {
    if (alias(a, null)) return alias(b, null);
    // a is non-null.
    if (alias(b, null)) return false;
    // a and b are non-null.
    if (alias(a.data, null)) return alias(b.data, null);
    // a.data is non-null.
    if (alias(b.data, null)) return false;
    // a.data and b.data are non-null.
    if (a.data.length != b.data.length) return false;
    for (int i = 0; i < a.data.length; ++i) {
      if (a.data[i] != b.data[i]) return false;
    }
    return true;
  }
  autounravel bool operator !=(Array_T a, Array_T b) {
    return !(a == b);
  }

  // Cast operators.
  autounravel Array_T operator cast(T[] x) { return Array_T(x); }
  autounravel T[] operator cast(Array_T x) { return x.data; }
}

Array_T wrap(T[] data, int hashElement(T x) = null) = Array_T;