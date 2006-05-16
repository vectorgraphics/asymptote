import TestLib;
StartTest("stat2");
struct T {
  int x;
  static void f(T t) {
    static void g(T t) {
      t.x=2;
    }
    g(t);
  }
}

T t=new T;
assert(t.x==0);
T.f(t);
assert(t.x==2);
EndTest();
