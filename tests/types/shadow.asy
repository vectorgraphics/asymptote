import TestLib;
StartTest("shadow");
int x = 1;
int getX() { return x; }
void setX(int value) { x=value; }

// Shadow x with another int, but x should still be in memory.
int x = 2;
assert(x==2);
assert(getX()==1);
x = 4;
assert(x==4);
assert(getX()==1);
setX(7);
assert(x==4);
assert(getX()==7);
EndTest();
