import TestLib;
StartTest("length");
assert(length("") == 0);
assert(length("abc") == 3);
assert(length("abcdef") == 6);
EndTest();
