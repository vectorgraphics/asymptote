import TestLib;
StartTest("erase");
string s = "abcdef";
Assert(erase(s,2,2) == "abef");
Assert(erase(s,-1,2) == "abcdef");
Assert(erase(s,7,1) == "abcdef");
Assert(erase(s,3,0) == "abcdef");
Assert(erase(s,5,2) == "abcde");
EndTest();
