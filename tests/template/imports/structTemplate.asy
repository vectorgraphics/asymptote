// Assumption: T is a struct with a static int `global`, a non-static int
// `local`, and an autounravel int `au`. The value of T.au is -1. Lib is a
// struct with a static string `testName`.
typedef import(T, Lib);
import TestLib;

StartTest('Accessing static testName');
Lib.testName;
EndTest();

StartTest(Lib.testName + ': new');
new T;
EndTest();

StartTest(Lib.testName + ': Unspecified Declaration');
T a;
EndTest();

StartTest(Lib.testName + ': Initializing to null');
T b = null;
EndTest();

StartTest(Lib.testName + ': Initializing to new');
T c = new T;
EndTest();

StartTest(Lib.testName + ': Access static member');
int d = T.global;
EndTest();

StartTest(Lib.testName + ': Access non-static member');
int e = c.local;
EndTest();

StartTest(Lib.testName + ': Access static member from instance');
int f = c.global;
EndTest();

StartTest(Lib.testName + ': Unraveling and accessing static member');
from T unravel global;
int g = global;
EndTest();

StartTest(Lib.testName + ': Access autounravel member');
assert(au == -1);
assert(T.au == -1);
assert(c.au == -1);
au = -2;
assert(au == -2);
assert(T.au == -2);
assert(c.au == -2);
T.au = -3;
assert(au == -3);
assert(T.au == -3);
assert(c.au == -3);
c.au = -4;
assert(au == -4);
assert(T.au == -4);
assert(c.au == -4);
au = -1;  // Reset for next test
EndTest();

StartTest(Lib.testName + ': Equality, inequality, and alias');
T h = new T;
assert(h == h);
assert(!(h == null));
assert(h != null);
assert(!(h != h));
assert(alias(h, h));
assert(!alias(h, null));
EndTest();