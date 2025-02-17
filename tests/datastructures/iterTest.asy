import TestLib;

StartTest('operator iter');

struct Iter_string {
  string get();
  void advance();
  bool valid();
}

struct Iterable_string {
  string[] a;
  void operator init(string[] a) {
    this.a = a;
  }
  Iter_string operator iter() {
    Iter_string it;
    int i = 0;
    it.get = new string() {
      return a[i];
    };
    it.advance = new void() {
      ++i;
    };
    it.valid = new bool() {
      return i < a.length;
    };
    return it;
  }
}

Iterable_string is = Iterable_string(new string[]{'a', 'b', 'c'});
for (var it = is.operator iter(); it.valid(); it.advance()) {
  assert(it.get() == 'a' || it.get() == 'b' || it.get() == 'c');
}

{
  // For loop with implicit variable type
  int count = 0;
  for (var s : is) {
    ++count;
    assert(s == 'a' || s == 'b' || s == 'c');
  }
  assert(count == 3);
}
{
  // For loop with explicit variable type
  int count = 0;
  for (string s : is) {
    ++count;
    assert(s == 'a' || s == 'b' || s == 'c');
  }
  assert(count == 3);
}
// Consider: iterate over enum via static operator iter()?



EndTest();