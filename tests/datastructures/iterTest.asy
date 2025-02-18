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

{
  // Test closure behavior
  struct ArrayIter {
    int[] a;
    int i;
    int get() {
      return a[i];
    }
    void advance() {
      ++i;
    }
    bool valid() {
      return i < a.length;
    }
    void operator init(int[] a) {
      this.a = a;
      i = 0;
    }
  }
  struct ArrayIterable {
    int[] a;
    void operator init(int[] a) {
      this.a = a;
    }
    ArrayIter operator iter() {
      return ArrayIter(a);
    }
  }
  using Function = int();
  ArrayIterable list = ArrayIterable(sequence(10));
  Function[] funcs;
  for (var i : list) {
    funcs.push(new int() {
      return i;
    });
  }
  for (int i = 0; i < 10; ++i) {
    assert(funcs[i]() == i);
  }
}
{
  // Implicitly cast a function to an array
  using Function = int(int);
  int[] operator cast(Function f) {
    return sequence(f, 10);
  }
  int f(int i) { return i + 17; }
  int f = 0;  // Cannot be cast to int[].
  int count = 0;
  for (int i : f) {
    assert(i == f(count));
    ++count;
  }
  assert(count == 10);
}


// Consider: iterate over enum via static operator iter()?



EndTest();