import TestLib;
StartTest("autounravel: struct declaration");
struct A {
  autounravel int x = -1;
}
assert(x == -1);
assert(A.x == -1);
x = -2;
assert(A.x == -2);
A.x = -3;
assert(x == -3);
EndTest();

StartTest("autounravel: typedef");
struct B {
  static struct C {
    autounravel int y = -1;
  }
}
typedef B.C C;
assert(y == -1);
assert(C.y == -1);
y = -2;
assert(C.y == -2);
C.y = -3;
assert(y == -3);
EndTest();

StartTest("autounravel: unravel *");
struct BBB {
  static struct CCC {
    autounravel int yyy = -1;
  }
}
unravel BBB;
assert(yyy == -1);
assert(CCC.yyy == -1);
yyy = -2;
assert(CCC.yyy == -2);
CCC.yyy = -3;
assert(yyy == -3);
EndTest();

StartTest("autounravel: unravel");
struct BB {
  static struct CC {
    autounravel int yy = -1;
  }
}
from BB unravel CC;
assert(yy == -1);
assert(CC.yy == -1);
yy = -2;
assert(CC.yy == -2);
CC.yy = -3;
assert(yy == -3);
EndTest();
