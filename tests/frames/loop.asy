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

// While loops
{
  int y = 7;
  int z = 0;
  while (z < 10) {
    ++z;
    continue;
    ++y;
  }
  assert(z == 10);
  assert(y == 7);
}

{
  int y = 7;
  int z = 0;
  while (z < 10) {
    void g() {}
    ++z;
    continue;
    ++y;
  }
  assert(z == 10);
  assert(y == 7);
}

{
  int y = 7;
  int z = 0;
  while (z < 10) {
    ++z;
    break;
    ++y;
  }
  assert(z == 1);
  assert(y == 7);
}

{
  int y = 7;
  int z = 0;
  while (z < 10) {
    void g() {}
    ++z;
    break;
    ++y;
  }
  assert(z == 1);
  assert(y == 7);
}


{
  int y = 7;
  int z = 0;
  while (z < 10) {
    ++z;
    continue;
    ++y;
  }
  assert(z == 10);
  assert(y == 7);
}

// Do loops
{
  int y = 7;
  int z = 0;
  do {
    void g() {}
    ++z;
    continue;
    ++y;
  } while (z < 10);
  assert(z == 10);
  assert(y == 7);
}

{
  int y = 7;
  int z = 0;
  do {
    ++z;
    break;
    ++y;
  } while (z < 10);
  assert(z == 1);
  assert(y == 7);
}

{
  int y = 7;
  int z = 0;
  do {
    void g() {}
    ++z;
    break;
    ++y;
  } while (z < 10);
  assert(z == 1);
  assert(y == 7);
}

{
  int x = 456;
  do { x = 123; } while (false);
  assert(x == 123);
}

{
  int x = 456;
  do { void g() {} x = 123; } while (false);
  assert(x == 123);
}


EndTest();
