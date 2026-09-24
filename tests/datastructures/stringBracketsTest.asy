import TestLib;

StartTest('string brackets');

string s = 'abc';
assert(s[0] == 'a');
assert(s[1] == 'b');
assert(s[2] == 'c');
assert(s[3] == '');
assert(s[-1] == 'c');
assert(s[-2] == 'b');
assert(s[-3] == 'a');
assert(s[-4] == '');

string f(int) = s.operator[];  // Check the type and signature of operator[].

assert(f(0) == 'a');

EndTest();