import TestLib;
StartTest("length");
Assert(length("") == 0);
Assert(length("abc") == 3);
Assert(length("abcdef") == 6);
EndTest();
