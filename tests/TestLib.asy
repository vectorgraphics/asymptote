void StartTest(string desc)
{ 
  write(stdout, "Testing " + desc + "...");
}

void EndTest()
{
  write("PASSED.");
}
