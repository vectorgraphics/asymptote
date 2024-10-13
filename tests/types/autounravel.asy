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

StartTest("autounravel: field is unraveled");
struct AAAA {
  static int z = -1;
}
struct BBBB {
  static struct CCCC {
    autounravel from AAAA unravel z as zz;
  }
}
from BBBB unravel CCCC;
assert(zz == -1);
assert(CCCC.zz == -1);
zz = -2;
assert(CCCC.zz == -2);
assert(AAAA.z == -2);
CCCC.zz = -3;
assert(zz == -3);
assert(AAAA.z == -3);
AAAA.z = -4;
assert(CCCC.zz == -4);
assert(zz == -4);
EndTest();

StartTest('autounravel: whole struct is unraveled');
struct A5 {
  static int z5 = -1;
}
struct B5 {
  static struct C5 {
    autounravel unravel A5;
  }
}
from B5 unravel C5;
assert(z5 == -1);
assert(C5.z5 == -1);
z5 = -2;
assert(C5.z5 == -2);
assert(A5.z5 == -2);
C5.z5 = -3;
assert(z5 == -3);
assert(A5.z5 == -3);
A5.z5 = -4;
assert(C5.z5 == -4);
assert(z5 == -4);
EndTest();

StartTest('autounravel: function');
struct B6 {
  static struct C6 {
    autounravel int y6() { return -1; }
  }
}
from B6 unravel C6;
assert(y6() == -1);
assert(C6.y6() == -1);
y6 = new int() { return -2; };
assert(C6.y6() == -2);
assert(y6() == -2);
C6.y6 = new int() { return -3; };
assert(y6() == -3);
assert(C6.y6() == -3);
EndTest();

StartTest('autounravel: multiple fields');
struct B7 {
  static struct C7 {
    autounravel int y7 = -1;
    autounravel int z7 = 1;
  }
}
from B7 unravel C7;
assert(y7 == -1);
assert(C7.y7 == -1);
assert(z7 == 1);
assert(C7.z7 == 1);
y7 = -2;
assert(C7.y7 == -2);
assert(z7 == 1);
assert(C7.z7 == 1);
C7.z7 = 2;
assert(y7 == -2);
assert(C7.y7 == -2);
assert(z7 == 2);
assert(C7.z7 == 2);
EndTest();