import TestLib;

typedef real F1(string);
typedef string F2(int);

real f1(string s) {
  return length(s);
}

F2 f2 = operator ecast;

StartTest("Function type parameters");

from 'template/imports/composeFunctions'(R=real, F1=F1, F2=F2, I=int) access
    compose;

real r = compose(f1, f2)(1234567890);
assert(r == 10);

EndTest();