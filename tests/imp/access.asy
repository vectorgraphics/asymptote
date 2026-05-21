import TestLib;

StartTest('dotted directories, no "as"');
access imp.imports.A;
assert(A.B.x == 4);
EndTest();