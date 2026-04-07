import TestLib;

StartTest('operator iter: empty iterable');
{
  struct EmptyIterable {
    // Iter_int defined in plain.asy.
    Iter_int operator iter() {
      Iter_int it;
      it.valid = new bool() { return false; };
      return it;
    }
  }

  EmptyIterable e;
  int count = 0;
  for (var x : e) {
    ++count;
  }
  assert(count == 0);
}
EndTest();

StartTest('operator iter: single element');
{
  struct SingleIterable {
    int value;
    void operator init(int v) { this.value = v; }
    // Iter_int defined in plain.asy.
    Iter_int operator iter() {
      bool done = false;
      Iter_int it;
      it.get = new int() { return value; };
      it.advance = new void() { done = true; };
      it.valid = new bool() { return !done; };
      return it;
    }
  }

  SingleIterable s = SingleIterable(42);
  int count = 0;
  int result = 0;
  for (var x : s) {
    result = x;
    ++count;
  }
  assert(count == 1);
  assert(result == 42);
}
EndTest();

StartTest('operator iter: nested for-each');
{
  // Two nested for-each loops over independent iterators of the same type.
  from collections.iter(T=int) access range, Iterable_T as Iterable_int;

  int[] a = {1, 2, 3};
  int[] b = {10, 20};

  int count = 0;
  for (int x : range(a)) {
    for (int y : range(b)) {
      ++count;
    }
  }
  assert(count == 6);  // 3 * 2
}
EndTest();

StartTest('operator iter: break from loop');
{
  from collections.iter(T=int) access range;

  int lastSeen = -1;
  for (int i : range(100)) {
    lastSeen = i;
    if (i == 5) break;
  }
  assert(lastSeen == 5);
}
EndTest();

StartTest('operator iter: continue in loop');
{
  from collections.iter(T=int) access range;

  int sum = 0;
  for (int i : range(10)) {
    if (i % 2 != 0) continue;
    sum += i;
  }
  // Sum of even numbers 0..9: 0+2+4+6+8 = 20
  assert(sum == 20);
}
EndTest();

StartTest('operator iter: for-each over array still works');
{
  // Verify that for-each over plain arrays continues to work alongside
  // the new struct-based operator iter protocol.
  int[] arr = {10, 20, 30, 40};
  int sum = 0;
  for (int x : arr) {
    sum += x;
  }
  assert(sum == 100);
}
EndTest();

StartTest('operator iter: multiple iterators from same object');
{
  // Each call to operator iter should produce an independent iterator.
  from collections.iter(T=int) access range, Iterable_T as Iterable_int;

  int[] data = {1, 2, 3, 4, 5};
  Iterable_int iterable = range(data);

  int sum1 = 0;
  for (int x : iterable) {
    sum1 += x;
  }

  int sum2 = 0;
  for (int x : iterable) {
    sum2 += x;
  }

  assert(sum1 == 15);
  assert(sum2 == 15);
}
EndTest();

StartTest('operator iter: var type inference');
{
  // Test that `var` correctly infers the type from get().
  from collections.iter(T=string) access Iter_T as Iter_string;

  struct StringIterable {
    string[] items;
    void operator init(string[] items) { this.items = items; }
    Iter_string operator iter() {
      int i = 0;
      Iter_string it;
      it.get = new string() { return items[i]; };
      it.advance = new void() { ++i; };
      it.valid = new bool() { return i < items.length; };
      return it;
    }
  }

  StringIterable si = StringIterable(new string[] {'hello', 'world'});
  string combined = '';
  for (var s : si) {
    combined += s;
  }
  assert(combined == 'helloworld');
}
EndTest();

StartTest('operator iter: locally-defined iterator struct');
{
  // Test that operator iter works with a locally-defined iterator struct
  // (structural matching) rather than requiring the library's Iter_T.
  // NOTE: This is NOT recommended style for users, but we want to ensure it
  // works as an edge case.
  struct MyIter {
    real get();
    void advance();
    bool valid();
  }

  struct Squares {
    int n;
    void operator init(int n) { this.n = n; }
    MyIter operator iter() {
      int i = 0;
      MyIter it;
      it.get = new real() { return i * i; };
      it.advance = new void() { ++i; };
      it.valid = new bool() { return i < n; };
      return it;
    }
  }

  Squares sq = Squares(5);
  real sum = 0;
  for (var x : sq) {
    sum += x;
  }
  // 0^2 + 1^2 + 2^2 + 3^2 + 4^2 = 0+1+4+9+16 = 30
  assert(sum == 30);
}
EndTest();
