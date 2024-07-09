// Assumption: T is a struct with a static int `global` and a non-static int
// `local`. Lib is a struct with a static string `testName`.
typedef import(T, Lib);
import TestLib;

StartTest('Accessing static testName');
Lib.testName;
EndTest();

StartTest(Lib.testName + ': new');
new T;
EndTest();

StartTest(Lib.testName + ': Unspecified Declaration');
T a;
EndTest();

StartTest(Lib.testName + ': Initializing to null');
T b = null;
EndTest();

StartTest(Lib.testName + ': Initializing to new');
T c = new T;
EndTest();

StartTest(Lib.testName + ': Access static member');
int d = T.global;
EndTest();

StartTest(Lib.testName + ': Access non-static member');
int e = c.local;
EndTest();

StartTest(Lib.testName + ': Access static member from instance');
int f = c.global;
EndTest();
