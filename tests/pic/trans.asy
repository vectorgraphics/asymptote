import TestLib;

StartTest("trans");

// Ensure the same test each time.
srand(3456);

pair randompair()
{
  return (unitrand(),unitrand());
}

path randombox()
{
  return box(randompair(), randompair());
}

// For now, only tests transforms which take axes to axes.
transform randomtrans()
{
  return rotate(90 * (rand() % 4)) * shift(unitrand(), unitrand());
}

real tolerance = 1e-4;

void testpic(int objs, int trans)
{
  path[] pp;
  picture orig;
  for (int i = 0; i < objs; ++i) {
    pp.push(randombox());
    fill(orig, pp[i]);
  }

  picture pic = orig;
  transform t = identity();
  for (int i = 0; i < trans; ++i)
  {
    transform tt = randomtrans();
    pic = tt * pic;
    t = tt * t;
  }

  pair m = pic.userMin2(), M = pic.userMax2();
  pair pm = min(t * pp), pM = max(t * pp);

  assert(abs(m-pm) < tolerance);
  assert(abs(M-pM) < tolerance);
}

for (int i = 0; i < 100; ++i)
  testpic(1,1);
for (int i = 0; i < 100; ++i)
  testpic(1,1);
for (int i = 0; i < 100; ++i)
  testpic(3,1);
for (int i = 0; i < 100; ++i)
  testpic(1,2);
for (int i = 0; i < 100; ++i)
  testpic(2,2);
for (int i = 0; i < 100; ++i)
  testpic(3,2);
for (int i = 0; i < 100; ++i)
  testpic(3,4);
for (int i = 0; i < 100; ++i)
  testpic(1,4);
for (int i = 0; i < 100; ++i)
  testpic(2,4);
for (int i = 0; i < 100; ++i)
  testpic(3,4);
for (int i = 0; i < 100; ++i)
  testpic(3,4);


EndTest();
