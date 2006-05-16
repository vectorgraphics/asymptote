import TestLib;
StartTest("rfind");
string s = "abcdefab";
assert(rfind(s,"cd") == 2);
assert(rfind(s,"cd",1) == -1);
assert(rfind(s,"ab") == 6);
assert(rfind(s,"ab",5) == 0);
EndTest();
