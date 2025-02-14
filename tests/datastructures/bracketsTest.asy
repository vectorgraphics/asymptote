import TestLib;

StartTest('brackets');

struct Foo {
  int x = 876;
  int y = 999;
  int default = -1010;

  int operator[](string key) {
    if (key == 'x')
      return x;
    if (key == 'y')
      return y;
    return default;
  }

  void operator[=](string key, int value) {
    if (key == 'x')
      x = value;
    if (key == 'y')
      y = value;
  }

  void setXY(int value) {
    this['x'] = this['y'] = value;
    assert(this['x'] == value);
    assert(this['y'] == value);
    ++this['y'];
    assert(this['y'] == value + 1);
    --this['y'];
    assert(this['y'] == value);
  }

  void reset() {
    x = 876;
    y = 999;
    default = -1010;
  }
}

Foo d;

assert(d['x'] == 876);
assert(d['y'] == 999);
assert(d['z'] == -1010);

d['x'] = 123;
d['y'] = 456;
d['z'] = 789;

assert(d['x'] == 123);
assert(d['y'] == 456);
assert(d['z'] == -1010);

d.setXY(32167);
assert(d['x'] == 32167);
assert(d['y'] == 32167);
assert(d['z'] == -1010);

d['x'] = d['y'] = d['z'] = 4567;
assert(d['x'] == 4567);
assert(d['y'] == 4567);
assert(d['z'] == -1010);

{
  d.reset();
  int initialX = d.x;
  // Check evaluation in self-operations.
  int count = 0;
  Foo func() {
    ++count;
    return d;
  }
  int count2 = 0;
  string x() {
    ++count2;
    return 'x';
  }

  func()[x()] += 111;
  assert(count == 1);
  assert(count2 == 1);
  assert(d['x'] == initialX+111);
  assert(func()[x()] == initialX+111);
  assert(count == 2);
  assert(count2 == 2);
}

{
  // Check settability of operator[] and operator[=].
  Foo e;
  e.operator[=]=new void(string key, int value) { };
  e['x'] = 123;
  assert(e['x'] == 876);
  e.operator[]=new int(string key) { return 0; };
  assert(e['x'] == 0);
}

{
  // Overload the object before the braces.
  d.reset();
  Foo d();
  d['x'] = 10191;
  assert(d['x'] == 10191);
}

{
  // Overload the value to be assigned.
  d.reset();
  int v = 451;
  int v() { return 10191; }
  d['x'] = v;
  assert(d['x'] == 451);
}

{
  d.reset();
  // Overload the key.
  string k = 'x';
  string k() { return 'y'; }
  assert(d[k] == 876);
  d[k] = 10191;
  assert(d['x'] == 10191);
}
{
  // Implicit casting on the key.
  d.reset();
  struct A {}
  int count = 0;
  string operator cast(A) {
    ++count;
    return 'x';
  }
  A k;
  assert(d[k] == 876);
  assert(count == 1);
  d[k] = 10191;
  assert(d['x'] == 10191);
  assert(count == 2);
}
{
  // Implicit casting on the value.
  d.reset();
  struct B {}
  int count = 0;
  int operator cast(B) {
    ++count;
    return 10191;
  }
  B v;
  d['x'] = d['y'] = d['z'] = v;
  assert(count == 1);
  assert(d['x'] == 10191);
  assert(d['y'] == 10191);
  assert(d['z'] == -1010);
}
{
  // Implicit casting when the value is a field of something other than a name.
  d.reset();
  struct A {}
  int count = 0;
  int operator cast(A) {
    ++count;
    return 10191;
  }
  struct B {
    A a;
  }
  d['x'] = d['y'] = d['z'] = (new B).a;
  assert(count == 1);
  assert(d['x'] == 10191);
  assert(d['y'] == 10191);
  assert(d['z'] == -1010);
}
{
  // Test the order of evaluation of the object, key, and value.
  int objectCount = 0;
  int keyCount = 0;
  int valueCount = 0;
  int setCount = 0;
  struct Bar {
    assert(++objectCount == 1);
    assert(keyCount == 0);
    assert(valueCount == 0);
    assert(setCount == 0);
    int operator[](string key) {
      assert(false);
      return 0;
    }
    void operator[=](string key, int value) {
      assert(objectCount == 1);
      assert(keyCount == 1);
      assert(valueCount == 1);
      assert(++setCount == 1);
    }
  }
  string getKey() {
    assert(objectCount == 1);
    assert(++keyCount == 1);
    assert(valueCount == 0);
    assert(setCount == 0);
    return '';
  }
  int getValue() {
    assert(objectCount == 1);
    assert(keyCount == 1);
    assert(++valueCount == 1);
    assert(setCount == 0);
    return 0;
  }
  (new Bar)[getKey()] = getValue();
}
{
  // Test the order of evaluation for a self-expression.
  int objectCount = 0;
  int keyCount = 0;
  int getCount = 0;
  int valueCount = 0;
  int setCount = 0;
  struct Bar {
    assert(++objectCount == 1);
    assert(keyCount == 0);
    assert(getCount == 0);
    assert(valueCount == 0);
    assert(setCount == 0);
    int operator[](string key) {
      assert(objectCount == 1);
      assert(keyCount == 1);
      assert(++getCount == 1);
      assert(valueCount == 0);
      assert(setCount == 0);
      return 0;
    }
    void operator[=](string key, int value) {
      assert(objectCount == 1);
      assert(keyCount == 1);
      assert(getCount == 1);
      assert(valueCount == 1);
      assert(++setCount == 1);
    }
  }
  string getKey() {
    assert(objectCount == 1);
    assert(++keyCount == 1);
    assert(getCount == 0);
    assert(valueCount == 0);
    assert(setCount == 0);
    return '';
  }
  int getValue() {
    assert(objectCount == 1);
    assert(keyCount == 1);
    assert(getCount == 1);
    assert(++valueCount == 1);
    assert(setCount == 0);
    return 0;
  }
  (new Bar)[getKey()] += getValue();
  assert(objectCount == 1);
  assert(keyCount == 1);
  assert(getCount == 1);
  assert(valueCount == 1);
  assert(setCount == 1);
}

EndTest();
