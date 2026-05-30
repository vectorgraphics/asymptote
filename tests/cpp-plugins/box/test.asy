// Phase 1 fixture: exercise Box class, methods, fields, method-as-value.
access "box" as b;

b.Box box = b.Box();
write(box.value);            // expect 42
write(box.size());           // expect 42

// Method-as-value: the design-doc form `Set_T.size = box.size;`
int f() = box.size;
write(f());                  // expect 42
