typedef import(R, F1, F2, I);

typedef R FF(I);

FF compose(F1 f1, F2 f2) {
  return new R(I i) {
    return f1(f2(i));
  };
}