import TestLib;
StartTest("order");

// Ordering tests.
int counter = 0;
int step(int n) { ++counter; assert(counter == n); return n; }
int[] stepArray(int n) { return new int[] {step(n)}; }
void reset() { counter = 0; }

{
  reset();
  assert(step(1) + step(2) == step(3)); step(4);
}

{
  reset();
  assert(step(1) == 0 || step(2) == 2); step(3);
}

{
  reset();
  step(1) == 0 && step(999) == step(456); step(2);
}

{
  reset();
  int x = step(1), y = step(2); step(3);
  int x = step(4), y = step(5); step(6);
}

{
  void f(int x, int y) {}

  reset();
  f(step(1), step(2)); step(3);
  reset();
  f(x=step(1), y=step(2)); step(3);
  reset();
  f(y=step(1), x=step(2)); step(3);
  reset();
  f(x=step(1), step(2)); step(3);
  reset();
  f(y=step(1), step(2)); step(3);
  reset();
  f(step(1), x=step(2)); step(3);
  reset();
  f(step(1), y=step(2)); step(3);
}

{
  void f(int x, int y ... int[] z) {}

  reset();
  f(step(1), step(2)); step(3);
  reset();
  f(x=step(1), y=step(2)); step(3);
  reset();
  f(y=step(1), x=step(2)); step(3);
  reset();
  f(x=step(1), step(2)); step(3);
  reset();
  f(y=step(1), step(2)); step(3);
  reset();
  f(step(1), x=step(2)); step(3);
  reset();
  f(step(1), y=step(2)); step(3);

  reset();
  f(step(1), step(2), step(3)); step(4);
  reset();
  f(x=step(1), y=step(2), step(3)); step(4);
  reset();
  f(y=step(1), x=step(2), step(3)); step(4);
  reset();
  f(x=step(1), step(2), step(3)); step(4);
  reset();
  f(y=step(1), step(2), step(3)); step(4);
  reset();
  f(step(1), x=step(2), step(3)); step(4);
  reset();
  f(step(1), y=step(2), step(3)); step(4);

  reset();
  f(step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(x=step(1), y=step(2), step(3), step(4)); step(5);
  reset();
  f(y=step(1), x=step(2), step(3), step(4)); step(5);
  reset();
  f(x=step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(y=step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(step(1), x=step(2), step(3), step(4)); step(5);
  reset();
  f(step(1), y=step(2), step(3), step(4)); step(5);

  reset();
  f(step(1), step(2), step(3)); step(4);
  reset();
  f(x=step(1), step(2), y=step(3)); step(4);
  reset();
  f(y=step(1), step(2), x=step(3)); step(4);
  reset();
  f(x=step(1), step(2), step(3)); step(4);
  reset();
  f(y=step(1), step(2), step(3)); step(4);
  reset();
  f(step(1), step(2), x=step(3)); step(4);
  reset();
  f(step(1), step(2), y=step(3)); step(4);

  reset();
  f(step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(x=step(1), step(2), y=step(3), step(4)); step(5);
  reset();
  f(y=step(1), step(2), x=step(3), step(4)); step(5);
  reset();
  f(x=step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(y=step(1), step(2), step(3), step(4)); step(5);
  reset();
  f(step(1), step(2), x=step(3), step(4)); step(5);
  reset();
  f(step(1), step(2), y=step(3), step(4)); step(5);

  reset();
  f(step(1), step(2), step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(x=step(1), step(2), y=step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(y=step(1), step(2), x=step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(x=step(1), step(2), step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(y=step(1), step(2), step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(step(1), step(2), x=step(3), step(4)... stepArray(5)); step(6);
  reset();
  f(step(1), step(2), y=step(3), step(4)... stepArray(5)); step(6);

  reset();
  f(...stepArray(1), x=step(2), y=step(3)); step(4);
  reset();
  f(...stepArray(1), y=step(2), x=step(3)); step(4);
  reset();
  f(step(1)...stepArray(2), x=step(3), y=step(4)); step(5);
  reset();
  f(step(1)...stepArray(2), y=step(3), x=step(4)); step(5);
  reset();
  f(step(1),step(2)...stepArray(3), y=step(4), x=step(5)); step(6);
  reset();
  f(step(1),step(2)...stepArray(3), y=step(4), x=step(5)); step(6);
  reset();
  f(step(1),step(2),x=step(3)...stepArray(4), y=step(5)); step(6);
  reset();
  f(step(1),step(2),x=step(3)...stepArray(4), y=step(5)); step(6);
}



// TODO: Add packing vs. casting tests.

EndTest();
