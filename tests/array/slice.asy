import TestLib;

StartTest("slice");

int[] x={0,1,2,3,4,5,6,7,8,9};

// Non-cyclic cases.
assert(all(x[:] == x));
assert(!alias(x[:],x));

assert(all(x[0:4] == new int[] {0,1,2,3} ));
assert(all(x[2:4] == new int[] {2,3} ));

assert(all(x[5:] == new int[] {5,6,7,8,9} ));
assert(all(x[:5] == new int[] {0,1,2,3,4} ));

assert(all(x[3:3] == new int[] {} ));
assert(all(x[3:4] == new int[] {3} ));
assert(all(x[98:99] == new int[] {} ));

assert(all(x[-100:] == x));

assert(all(x[5:3] == new int[] {} ));
assert(all(x[105:3] == new int[] {} ));

assert(x[:].cyclicflag == false);
assert(x[2:].cyclicflag == false);
assert(x[:7].cyclicflag == false);
assert(x[3:3].cyclicflag == false);
assert(x[2:9].cyclicflag == false);
assert(x[9:2].cyclicflag == false);

// Cyclic cases
x.cyclic(true);

assert(all(x[:] == new int[] {0,1,2,3,4,5,6,7,8,9} ));
assert(all(x[0:4] == new int[] {0,1,2,3} ));
assert(all(x[2:4] == new int[] {2,3} ));

assert(all(x[5:] == new int[] {5,6,7,8,9} ));
assert(all(x[:5] == new int[] {0,1,2,3,4} ));

assert(all(x[3:3] == new int[] {} ));
assert(all(x[3:4] == new int[] {3} ));

assert(all(x[-1:1] == new int[] {9,0} ));
assert(all(x[9:11] == new int[] {9,0} ));
assert(all(x[9:21] == new int[] {9,0,1,2,3,4,5,6,7,8,9,0} ));
assert(all(x[-15:15] == new int[] {5,6,7,8,9,0,1,2,3,4,5,6,7,8,9,
                                   0,1,2,3,4,5,6,7,8,9,0,1,2,3,4}));
assert(all(x[6728:6729] == new int[] {8} ));
assert(all(x[-6729:-6728] == new int[] {1} ));

assert(all(x[5:3] == new int[] {} ));
assert(all(x[105:3] == new int[] {} ));

assert(x[:].cyclicflag == false);
assert(x[2:].cyclicflag == false);
assert(x[:7].cyclicflag == false);
assert(x[3:3].cyclicflag == false);
assert(x[2:9].cyclicflag == false);
assert(x[9:2].cyclicflag == false);
assert(x[5:100].cyclicflag == false);

EndTest();
