import TestLib;
import math;

string[] a={"bob","alice","pete","alice"};
string[] b={"alice","alice","bob","pete"};

StartTest("sort");

assert(all(sort(a) == b));

EndTest();

StartTest("search");

assert(search(b,"a") == -1);
assert(search(b,"bob") == 2);
assert(search(b,"z") == b.length-1);

EndTest();

StartTest("sort2");

string[][] a={{"bob","9"},{"alice","5"},{"pete","7"},{"alice","4"}};
string[][] b={{"alice","4"},{"alice","5"},{"bob","9"},{"pete","7"}};

assert(sort(a) == b);

EndTest();

pair[] a={(2,1),(0,0),(1,1),(1,0)};
pair[] b={(0,0),(1,0),(1,1),(2,1)};

StartTest("lexicographical sort");
assert(all(sort(a,lexorder) == b));
EndTest();

StartTest("lexicographical search");
assert(search(b,(1,0),lexorder) == 1);
EndTest();
