// Real arithmetic.

import TestLib;
StartTest("real error");
Assert((1.0-1.0) < realEpsilon);
EndTest();
StartTest("real addition");
Assert((1.0+1.0) == (2.0));
EndTest();
StartTest("real subtraction");
Assert((2.0-1.0) == (1.0));
EndTest();
StartTest("real multiplication");
Assert((2.0*2.0) == (4.0));
EndTest();
StartTest("real division");
Assert((4.0/2.0) == (2.0));
EndTest();
