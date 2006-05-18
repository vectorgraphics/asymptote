bool close(real a, real b) 
{
  real norm=(b == 0) ? 1 : max(abs(a),abs(b));
  return abs(a-b) <= 10*realEpsilon*norm;
}

void StartTest(string desc)
{ 
  write(stdout, "Testing " + desc + "...");
}

void EndTest()
{
  write("PASSED.");
}
