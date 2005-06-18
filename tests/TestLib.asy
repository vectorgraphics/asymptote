bool success = false;

void StartTest(string desc)
{ write(stdout, "Testing " + desc + "...");
  success = true; }

void Assert(bool test)
{
  success = success && test;
}

void Pass()
{ write('PASSED.'); }

void Fail()
{ write('FAILED.'); }

void EndTest()
{
  if (success) Pass();
  else Fail();
}
