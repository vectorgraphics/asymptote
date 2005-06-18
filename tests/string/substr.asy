import TestLib;
StartTest("substr");
string s = "abcdef";
string sub = substr(s,2,2);
Assert(sub == "cd");
EndTest();
