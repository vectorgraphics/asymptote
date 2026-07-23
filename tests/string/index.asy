import TestLib;
StartTest("index");
string s = "abcdef";
assert(s[0] == "a");
assert(s[3] == "d");
assert(s[5] == "f");
assert(s[-1] == "f");
assert(s[-3] == "d");
assert(s[-6] == "a");
assert(s[6] == "");
assert(s[-7] == "");
string t = "";
assert(t[0] == "");
assert(t[-1] == "");
EndTest();

