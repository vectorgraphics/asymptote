import TestLib;

StartTest('recordOpEqAsValue: == and != as function values');
{
  struct R {
    int x;
  }
  R a = new R; a.x = 1;
  R b = new R; b.x = 1;
  R c = a;

  // Coerce builtin record == and != to concrete function types.
  bool equiv(R, R) = operator ==;
  bool different(R, R) = operator !=;

  // Builtin == is identity-based for records.
  assert(!equiv(a, b));
  assert(equiv(a, c));
  assert(equiv(a, a));
  assert(different(a, b));
  assert(!different(a, c));

  // Null handling.
  assert(!equiv(a, null));
  assert(equiv((R)null, null));
  assert(different(a, null));
  assert(!different((R)null, null));
}
EndTest();
