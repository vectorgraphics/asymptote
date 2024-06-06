access "template/imports/A"(T=int) as a;
unravel a;
access "template/imports/B"(T=A) as b;
unravel b;

import TestLib;

StartTest("init");
A c;
EndTest();

StartTest("new");
struct X {
  static struct A {
    int x=1;
  }
}

X x;

access "template/imports/C"(T=x.A) as p;

EndTest();
