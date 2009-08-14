import TestLib;
import math;

StartTest("delete");

int[] a=sequence(4);
a.delete(2);
assert(all(a == new int[] {0,1,3}));

int[] a=sequence(4);
a.delete(0,2);
assert(all(a == new int[] {3}));

int[] a=sequence(4);
a.delete(1,2);
assert(all(a == new int[] {0,3}));

int[] a=sequence(4);
a.delete(2,2);
assert(all(a == new int[] {0,1,3}));

int[] a=sequence(4);
a.delete(2,3);
assert(all(a == new int[] {0,1}));

int[] a=sequence(4);
a.cyclic=true;
a.delete(2,3);
assert(all(a == new int[] {0,1}));

int[] a=sequence(4);
a.cyclic=true;
a.delete(2,4);
assert(all(a == new int[] {1}));

int[] a=sequence(4);
a.cyclic=true;
a.delete(3,1);
assert(all(a == new int[] {2}));

EndTest();
