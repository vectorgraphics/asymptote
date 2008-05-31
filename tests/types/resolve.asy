import TestLib;
StartTest("resolve");
struct A {} struct B {} struct C {}

int f(B, real) { return 1; }
int f(C, int) { return 2; }
B operator cast(A) { return new B; }

assert(f(new A, 3) == 1);
C operator cast(A) { return new C; }
assert(f(new A, 3) == 2);
EndTest();
