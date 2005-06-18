import TestLib;
StartTest("find");
string s = "abcdefab";
Assert(find(s,"cd") == 2);
Assert(find(s,"cd",3) == -1);
Assert(find(s,"ab") == 0);
Assert(find(s,"ab",1) == 6);
EndTest();
