import TestLib;
StartTest('range vs sequence');

assert(all(sequence(10) == (int[])range(10)));
assert(all(sequence(10, 20) == (int[])range(10, 20)));
assert(all(sequence(10, 20, 2) == (int[])range(10, 20, 2)));
assert(all(sequence(10, 20, -2) == (int[])range(10, 20, -2)));
assert(all(sequence(20, 10, -2) == (int[])range(20, 10, -2)));


EndTest();