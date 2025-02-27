import TestLib;

StartTest('int.hash');
{
  int x = 3;
  assert(x.hash() >= 0);
  assert(x.hash() == (3).hash());
  for (int i = 0; i < 1000; ++i) {
    assert(i.hash() >= 0);
  }
  // The hash should be roughly uniformly distributed in the space of 62-bit
  // integers.
  // This assertion will fail on roughly 1 in 2^32 runs.
  assert(x.hash() > 2^30,
         'Probabilistic test failed. Chance of spurious failure is roughly 1 '
         'in 2^32.');
  // This assertion will fail on roughly 1 in 2^62 runs.
  assert(x.hash() != (4).hash());
}
EndTest();

StartTest('string.hash');
{
  string s = 'hello';
  assert(s.hash() >= 0);
  assert(s.hash() == ('hello').hash());

  // The hash should be roughly uniformly distributed in the space of 62-bit
  // integers.
  // This assertion will fail on roughly 1 in 2^32 runs.
  assert(s.hash() > 2^30,
         'Probabilistic test failed. Chance of spurious failure is roughly 1 '
         'in 2^32.');
  // This assertion will fail on roughly 1 in 2^62 runs.
  assert(s.hash() != ('world').hash()
         'Probabilistic test failed. Chance of spurious failure is roughly 1 '
         'in 2^32.');
}
EndTest();

StartTest('real.hash');
{
  real ONE = 1.0;
  assert(ONE.hash() >= 0);
  assert(ONE.hash() == (1.0).hash());
  // The hash should be roughly uniformly distributed in the space of 62-bit
  // integers.
  // This assertion will fail on roughly 1 in 2^32 runs.
  assert(ONE.hash() > 2^30,
         'Probabilistic test failed. Chance of spurious failure is roughly 1 '
         'in 2^32.');
  // This assertion will fail on roughly 1 in 2^62 runs.
  assert(ONE.hash() != (1.0 + 1e-15).hash());
  // This assertion will fail on roughly 1 in 2^62 runs.
  assert((1.0).hash() != (1).hash());
}
EndTest();

StartTest('hash(int[])');
{
  int[] a = {1, 2, 3};
  assert(hash(a) >= 0);
  assert(hash(a) == hash(new int[] {1, 2, 3}));
  // The hash should be roughly uniformly distributed in the space of 62-bit
  // integers.
  // This assertion will fail on roughly 1 in 2^32 runs.
  assert(hash(a) > 2^30,
         'Probabilistic test failed. Chance of spurious failure is roughly 1 '
         'in 2^32.');
  // This assertion will fail on roughly 1 in 2^62 runs.
  assert(hash(a) != hash(new int[] {1, 2, 4}));
}
EndTest();