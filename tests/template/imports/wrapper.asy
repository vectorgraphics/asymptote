typedef import(T);

struct Wrapper_T {
  T t;
  void operator init(T t) {
    this.t = t;
  }
}

Wrapper_T wrap(T t) {
  return Wrapper_T(t);
}

bool operator == (Wrapper_T a, Wrapper_T b) {
  return a.t == b.t;
}