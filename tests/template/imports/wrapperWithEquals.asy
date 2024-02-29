typedef import(T);

access "template/imports/wrapper"(T=T) as wrapper;
unravel wrapper;

bool operator == (Wrapper_T a, Wrapper_T b) {
  return a.t == b.t;
}
bool operator != (Wrapper_T a, Wrapper_T b) {
  return a.t != b.t;
}