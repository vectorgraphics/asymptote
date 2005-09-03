import TestLib;
StartTest("stat");
struct T {
  int x;
  static void f(T t) {
    t.x=2;
  }
}

T t=new T;
Assert(t.x==0);
T.f(t);
Assert(t.x==2);
EndTest();
