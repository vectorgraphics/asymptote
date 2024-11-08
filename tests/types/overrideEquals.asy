import TestLib;

StartTest("overrideEquals: external");
{
  struct Outer {
    static struct Inner {}
  }
  from Outer unravel Inner;
  Inner a = new Inner;
  assert(!(a == null));
  bool operator ==(Inner a, Inner b) {
    return true;
  }
  assert(a == null);
  typedef Inner _;  // Force autounravel to rerun.
  // Even if autounravel is rerun, it should not shadow something that was
  // already shadowed.
  assert(a == null);
}
EndTest();

StartTest('overrideEquals: internal');
{
  struct Outer {
    static struct Inner {
      autounravel bool operator ==(Inner a, Inner b) {
        return true;
      }
    }
  }
  from Outer unravel Inner;
  Inner a = new Inner;
  assert(a == null);  // Use the overridden operator.
}
EndTest();