import TestLib;

StartTest('join');
string[] a={'ab','cd','ef'};
assert(join(a) == 'abcdef');
assert(join(a,', ') == 'ab, cd, ef');
assert(join(a,delimiter='-') == 'ab-cd-ef');
assert(join(new string[] {'x'},',') == 'x');
assert(join(new string[],',') == '');
assert(join(new string[] {'','',''},',') == ',,');
assert(join(new string[] {'a','','b'},'') == 'ab');
EndTest();

StartTest('join_split');
string s='one two  three';
assert(join(split(s,' '),' ') == s);
assert(join(split('a,b,,c',','),',') == 'a,b,,c');
EndTest();

StartTest('operator +(...string[])');
assert(operator +(...a) == 'abcdef');
assert(operator +('x','y','z') == 'xyz');
assert(operator +(...new string[]) == '');
EndTest();
