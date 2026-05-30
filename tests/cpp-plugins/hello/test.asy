// Phase 0 fixture: verify that a C++ plugin can be loaded and that its
// exported functions are callable from asy.
from hello access hello, greet, sum;

write(hello());
write(greet("world"));
write(sum(40, 2));
