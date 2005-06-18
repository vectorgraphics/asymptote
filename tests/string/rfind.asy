import TestLib;
StartTest("rfind");
string s = "abcdefab";
Assert(rfind(s,"cd") == 2);
Assert(rfind(s,"cd",1) == -1);
Assert(rfind(s,"ab") == 6);
Assert(rfind(s,"ab",5) == 0);
EndTest();
