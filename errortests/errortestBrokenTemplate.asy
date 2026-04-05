// File to be incorrectly imported from errortest.asy.
typedef import(A, B, C);
assert(false);
// typedef import gives an error if not the first line
typedef import(A, B, C);
