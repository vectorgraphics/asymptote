import TestLib;
StartTest("loop");

int f();
for (int i=0; i<10; ++i) { 
  int x=i;
  for (int j=0; j<10; ++j) {
    int y=j;
    if (i==5 && j==7) {
      f = new int () { return x*y; };
    }
  }
}
assert(f()==35);

int f();

for (int i=0; i<10; ++i) { 
  int x=i;
  for (int j=0; j<10; ++j) {
    int y=j;
    if (i==5 && j==7) {
      f = new int () { return i*y; };
    }
  }
}
assert(f()==70);

EndTest();
