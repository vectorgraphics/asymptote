import TestLib;

StartTest('multiple_imports');
struct A {int x=1;}
access template.imports.C(T=A) as p;
assert(p.global == 17);
p.global = 42;
access template.imports.C(T=A) as q;
assert(q.global == 42);
EndTest();

StartTest('import_in_function');
struct B {int x=1;}
void f(int expected, int newValue) {
  // Importing inside a function is not recommended practice, but it should
  // work.
  access template.imports.C(T=B) as p;
  assert(p.global == expected);
  p.global = newValue;
}
f(17, 23);
f(23, 27);
access template.imports.C(T=B) as p;
assert(p.global == 27);
EndTest();
