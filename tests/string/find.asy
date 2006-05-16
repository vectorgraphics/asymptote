import TestLib;
StartTest("find");
string s = "abcdefab";
assert(find(s,"cd") == 2);
assert(find(s,"cd",3) == -1);
assert(find(s,"ab") == 0);
assert(find(s,"ab",1) == 6);
EndTest();
