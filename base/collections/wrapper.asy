typedef import(T);

struct Wrapped_T {
  T t;
  void operator init(T t) {
    this.t = t;
  }
  autounravel bool operator ==(Wrapped_T a, Wrapped_T b) {
    return a.t == b.t;
  }
  autounravel bool operator !=(Wrapped_T a, Wrapped_T b) {
    // Let's not assume that != was overloaded.
    return !(a.t == b.t);
  }
}

Wrapped_T wrap(T t) {
  return Wrapped_T(t);
}