bool close(pair a, pair b) 
{
  real norm=(b == 0) ? 1 : max(abs(a),abs(b));
  return abs(a-b) <= 100*realEpsilon*norm;
}

void StartTest(string desc)
{ 
  write("Testing " + desc + "...",flush);
}

void EndTest()
{
  write("PASSED.");
}
