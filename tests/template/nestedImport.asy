import TestLib;

StartTest('nested_import');
struct A { int x = 1; }
access 'template/imports/Cpass'(T=A) as module;
assert(module.global == 17);
EndTest();