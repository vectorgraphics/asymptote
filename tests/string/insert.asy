import TestLib;
StartTest("insert");
string sub = insert("abef",2,"cd");
Assert(sub == "abcdef");
EndTest();
