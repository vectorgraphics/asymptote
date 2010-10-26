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

{
  int y = 3;
  int z = 0;
  for (int i = 0; i < 7; ++i)
  {
    ++z;
    continue;
    y = 4;
  }
  assert(y == 3);
  assert(z == 7);
}
{
  int y = 3;
  int z = 0;
  for (int i = 0; i < 7; ++i)
  {
    ++z;
    break;
    y = 4;
  }
  assert(y == 3);
  assert(z == 1);
}
{
  int y = 3;
  int z = 0;
  for (int i = 0; i < 7; ++i)
  {
    void g() {}
    ++z;
    continue;
    y = 4;
  }
  assert(y == 3);
  assert(z == 7);
}
{
  int y = 3;
  int z = 0;
  for (int i = 0; i < 7; ++i)
  {
    void g() {}
    ++z;
    break;
    y = 4;
  }
  assert(y == 3);
  assert(z == 1);
}


EndTest();
