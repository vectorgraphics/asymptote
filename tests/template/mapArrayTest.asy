import TestLib;

StartTest("mapArray");

from mapArray(Src=real, Dst=string) access map;

real[] a = {1.0, 1.5, 2.5, -3.14};
string[] b = map(new string(real x) {return (string)x;},
                 a);

assert(all(b == new string[] {"1", "1.5", "2.5", "-3.14"}));

EndTest();