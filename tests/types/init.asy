import TestLib;
StartTest("init");
int operator init() { return 7; }
int x;
Assert(x==7);
EndTest();
