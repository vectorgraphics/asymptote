// Integer arithmetic.
import TestLib;
StartTest("integer addition");
assert(1+1==2);
EndTest();
StartTest("integer subtraction");
assert(2-1==1);
EndTest();
StartTest("integer multiplication");
assert(2*2==4);
EndTest();
StartTest("integer division");
assert(4/2==2);
EndTest();
StartTest("integer self ops");
{ int x = 3; assert(++x==4); assert(x==4); }
{ int x = 3; assert(--x==2); assert(x==2); }
{ int x = 3; assert((x+=7) == 10); assert(x==10); }
{ int x = 3; assert((x-=7) == -4); assert(x==-4); }
{ int x = 3; assert((x*=7) == 21); assert(x==21); }
{ int x = 10; assert((x%=4) == 2); assert(x==2); }
EndTest();

