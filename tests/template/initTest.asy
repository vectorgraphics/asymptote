import TestLib;

StartTest("init");

struct X {
  struct A {
  }
}

X x;

access "template/imports/p"(T=x.A) as p;

EndTest();
