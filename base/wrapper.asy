typedef import(T);

struct Wrapper_T {
  T t;
  void operator init(T t) {
    this.t = t;
  }
  autounravel bool operator ==(Wrapper_T a, Wrapper_T b) {
    // NOTE: This won't compile if T is an array type since == is
    // vectorized for arrays.
    return a.t == b.t;
  }
  autounravel bool operator !=(Wrapper_T a, Wrapper_T b) {
    // Let's not assume that != was overloaded.
    return !(a.t == b.t);
  }
  autounravel bool alias(Wrapper_T, Wrapper_T) = alias;
}

Wrapper_T wrap(T t) {
  return Wrapper_T(t);
}