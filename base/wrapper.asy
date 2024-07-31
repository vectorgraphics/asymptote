typedef import(T);

struct Wrapper_T {
  T t;
  void operator init(T t) {
    this.t = t;
  }
  if type (bool operator ==(T, T)) unravel {
    autounravel bool operator ==(Wrapper_T a, Wrapper_T b) {
      return a.t == b.t;
    }
    autounravel bool operator !=(Wrapper_T a, Wrapper_T b) {
      // Let's not assume that != was overloaded.
      return !(a.t == b.t);
    }
  }
  if type (bool operator <(T, T)) unravel {
    autounravel bool operator <(Wrapper_T a, Wrapper_T b) {
      return a.t < b.t;
    }
  }
  if type (bool operator <=(T, T)) unravel {
    autounravel bool operator <=(Wrapper_T a, Wrapper_T b) {
      return a.t <= b.t;
    }
  }
  if type (bool operator >(T, T)) unravel {
    autounravel bool operator >(Wrapper_T a, Wrapper_T b) {
      return a.t > b.t;
    }
  }
  if type (bool operator >=(T, T)) unravel {
    autounravel bool operator >=(Wrapper_T a, Wrapper_T b) {
      return a.t >= b.t;
    }
  }

  autounravel bool alias(Wrapper_T, Wrapper_T) = alias;
}

Wrapper_T wrap(T t) {
  return Wrapper_T(t);
}