import TestLib;
StartTest('enumerate');

from collections.enumerate(T=string) access
    enumerate,
    Iterable_Pair_int_T as Iterable_Pair_int_string,
    Iterable_T as Iterable_string;

from collections.iter(T=string) access Iterable;

string[] strings = {'a', 'b', 'c'};

{
  // enumerate over iterable
  bool[] triggered = array(strings.length, false);

  for (var is : enumerate(Iterable(strings))) {
    int i = is.k;
    string s = is.v;
    assert(s == strings[i]);
    for (int j = 0; j < i; ++j) {
      assert(triggered[j]);
    }
    for (int j = i; j < strings.length; ++j) {
      assert(!triggered[j]);
    }
    assert(!triggered[i]);
    triggered[i] = true;
  }

  assert(all(triggered));
}

{
  // enumerate over array
  bool[] triggered = array(strings.length, false);

  for (var is : enumerate(strings)) {
    int i = is.k;
    string s = is.v;
    assert(s == strings[i]);
    for (int j = 0; j < i; ++j) {
      assert(triggered[j]);
    }
    for (int j = i; j < strings.length; ++j) {
      assert(!triggered[j]);
    }
    assert(!triggered[i]);
    triggered[i] = true;
  }

  assert(all(triggered));
}

EndTest();