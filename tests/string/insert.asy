import TestLib;
StartTest("insert");
string sub = insert("abef",2,"cd");
assert(sub == "abcdef");
EndTest();
