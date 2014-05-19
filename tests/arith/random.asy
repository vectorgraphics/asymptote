import TestLib;
StartTest("random");
bool bit32=false;
for(int i=0; i < 1000; ++i) {
  real x=unitrand();
  if(x > 0.5) bit32=true;
  assert(x >= 0.0 && x <= 1.0);
}
assert(bit32);

EndTest();
