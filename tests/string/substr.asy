import TestLib;
StartTest("substr");
string s = "abcdef";
string sub = substr(s,2,2);
assert(sub == "cd");
EndTest();
