import TestLib;
StartTest("stat");
struct T {
  int x;
  static void f(T t) {
    t.x=2;
  }
}

T t=new T;
assert(t.x==0);
T.f(t);
assert(t.x==2);
EndTest();
