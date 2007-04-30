import TestLib;
StartTest("spec");

// Test if the cycle keyword can be used in different contexts.
{
  int operator cast(cycleToken) {
    return 55;
  }
  int x=cycle;
  assert(x==55);
}
EndTest();

