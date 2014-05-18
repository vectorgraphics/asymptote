import TestLib;
StartTest("random");
for(int i=0; i < 1000; ++i) {
  real x=unitrand();
  assert(x >= 0.0 && x <= 1.0);
}
EndTest();
