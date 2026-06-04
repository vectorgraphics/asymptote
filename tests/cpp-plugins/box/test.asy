// Phase 1 fixture: exercise Box class, methods, fields, method-as-value.
access "box" as b;

b.Box box = b.Box();
assert(box.value == 42, 'Box.value');
assert(box.size() == 42, 'Box.size()');

// Method-as-value: the design-doc form `Set_T.size = box.size;`
int f() = box.size;
assert(f() == 42, 'Box.size as a value');
