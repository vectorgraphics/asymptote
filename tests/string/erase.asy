import TestLib;
StartTest("erase");
string s = "abcdef";
assert(erase(s,2,2) == "abef");
assert(erase(s,-1,2) == "abcdef");
assert(erase(s,7,1) == "abcdef");
assert(erase(s,3,0) == "abcdef");
assert(erase(s,5,2) == "abcde");
EndTest();
