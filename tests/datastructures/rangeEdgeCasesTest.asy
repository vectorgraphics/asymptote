import TestLib;

StartTest('range: empty ranges');
{
  // range(0) should produce an empty sequence
  int count = 0;
  for (int i : range(0)) {
    ++count;
  }
  assert(count == 0);

  // range with reversed bounds and positive skip should be empty
  count = 0;
  for (int i : range(10, 5)) {
    ++count;
  }
  assert(count == 0);

  // range with forward bounds and negative skip should be empty
  count = 0;
  for (int i : range(5, 10, -1)) {
    ++count;
  }
  assert(count == 0);

  // range with skip=0 should be empty (never valid)
  count = 0;
  for (int i : range(0, 10, 0)) {
    ++count;
  }
  assert(count == 0);
}
EndTest();

StartTest('range: single element');
{
  int[] result;
  for (int i : range(1)) {
    result.push(i);
  }
  assert(result.length == 1);
  assert(result[0] == 0);
}
EndTest();

StartTest('range: negative skip');
{
  int[] result;
  for (int i : range(10, 0, -2)) {
    result.push(i);
  }
  assert(result.length == 6);
  assert(result[0] == 10);
  assert(result[1] == 8);
  assert(result[2] == 6);
  assert(result[3] == 4);
  assert(result[4] == 2);
  assert(result[5] == 0);
}
EndTest();

StartTest('range: large skip');
{
  int[] result;
  for (int i : range(0, 100, 30)) {
    result.push(i);
  }
  assert(result.length == 4);
  assert(result[0] == 0);
  assert(result[1] == 30);
  assert(result[2] == 60);
  assert(result[3] == 90);
}
EndTest();

StartTest('range: negative values');
{
  int[] result;
  for (int i : range(-5, -1)) {
    result.push(i);
  }
  assert(result.length == 5);
  assert(result[0] == -5);
  assert(result[4] == -1);
}
EndTest();

StartTest('range: cast to array');
{
  assert(all(sequence(5) == (int[])range(5)));
  assert(all(sequence(3, 7) == (int[])range(3, 7)));
  assert(all(sequence(0, 10, 3) == (int[])range(0, 10, 3)));
  assert(all(sequence(10, 0, -3) == (int[])range(10, 0, -3)));
}
EndTest();

StartTest('range: re-iterate');
{
  // An iterable should be re-iterable (each call to operator iter gives
  // a fresh iterator).
  var r = range(5);
  int[] first;
  for (int i : r) {
    first.push(i);
  }
  int[] second;
  for (int i : r) {
    second.push(i);
  }
  assert(all(first == second));
  assert(first.length == 5);
}
EndTest();
