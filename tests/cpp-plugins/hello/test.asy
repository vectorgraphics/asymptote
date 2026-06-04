// Phase 0 fixture: verify that a C++ plugin can be loaded and that its
// exported functions are callable from asy.
from hello access hello, greet, sum;

assert(hello() == 'hi');
assert(greet('world') == 'hello, world');
assert(sum(40, 2) == 42);
