// Real arithmetic.

import TestLib;
StartTest("real error");
assert((1.0-1.0) < realEpsilon);
EndTest();
StartTest("real addition");
assert((1.0+1.0) == (2.0));
EndTest();
StartTest("real subtraction");
assert((2.0-1.0) == (1.0));
EndTest();
StartTest("real multiplication");
assert((2.0*2.0) == (4.0));
EndTest();
StartTest("real division");
assert((4.0/2.0) == (2.0));
EndTest();
StartTest("real mod");
assert((12.0%5.0) == (2.0));
assert((-12.0%5.0) == (3.0));
assert((12.0%-5.0) == (-3.0));
assert((-12.0%-5.0) == (-2.0));
EndTest();
