// Heterogeneous write() tests.
import TestLib;

string tmpfile = "write_test_tmp.txt";

void doWrite(... var[] items)
{
  file f = output(tmpfile);
  write(f, ...items);
  close(f);
}

string doRead()
{
  file g = input(tmpfile);
  string s = g;
  close(g);
  return s;
}

// Write a single value through a `var`-typed parameter.  This is the path that
// used to segfault: a raw value entering a var slot (a var parameter fed a
// literal) must be wrapped in a tagged_var so write() can dispatch on it.
void writeVarValue(var x)
{
  file f = output(tmpfile);
  write(f, x);
  close(f);
}

StartTest("write single int");
doWrite(42, endl);
assert(doRead() == "42");
EndTest();

StartTest("write single real");
doWrite(3.14, endl);
assert(doRead() == "3.14");
EndTest();

StartTest("write single string");
doWrite("hello", endl);
assert(doRead() == "hello");
EndTest();

StartTest("write two ints");
doWrite(42, 7, endl);
assert(doRead() == "42" + "	" + "7");
EndTest();

StartTest("write int and real");
doWrite(42, 3.14, endl);
assert(doRead() == "42" + "	" + "3.14");
EndTest();

StartTest("write real and pair");
doWrite(1.5, (1,2), endl);
assert(doRead() == "1.5" + "	" + "(1,2)");
EndTest();

StartTest("write three heterogeneous");
doWrite(42, 3.14, "end", endl);
assert(doRead() == "42" + "	" + "3.14" + "	" + "end");
EndTest();

StartTest("write label and int");
doWrite("x =", 42, endl);
assert(doRead() == "x =42");
EndTest();

StartTest("write label and two values");
doWrite("x =", 42, 7, endl);
assert(doRead() == "x =42" + "	" + "7");
EndTest();

StartTest("write two strings");
doWrite("a", "b", endl);
assert(doRead() == "ab");
EndTest();

StartTest("write two pairs");
doWrite((1,2), (3,4), endl);
assert(doRead() == "(1,2)" + "	" + "(3,4)");
EndTest();

StartTest("write two bools");
doWrite(true, false, endl);
assert(doRead() == "true " + "	" + "false ");
EndTest();

StartTest("write label and int (struct method)");
Label L = "hi";
doWrite(L, 42, endl);
assert(doRead() == "\"hi\"" + "	" + "42");
EndTest();

StartTest("write two labels (struct method)");
Label A = "first";
Label B = "second";
doWrite(A, B, endl);
assert(doRead() == "\"first\"" + "	" + "\"second\"");
EndTest();

StartTest("write string prefix and label");
Label L = "value";
doWrite("key", L, endl);
assert(doRead() == "key\"value\"");
EndTest();

StartTest("write with flush suffix (no newline)");
{
  doWrite(42, flush);
  file g = input(tmpfile);
  string s; s = g; assert(s == "42");
  close(g);
}
EndTest();

StartTest("write with endl suffix (newline)");
{
  doWrite(42, endl);
  file g = input(tmpfile);
  string s; s = g; assert(s == "42");
  close(g);
}
EndTest();

StartTest("write multiple lines");
{
  file f = output(tmpfile);
  write(f, 1, 2, endl);
  write(f, 3, 4, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1" + "	" + "2");
  s = g; assert(s == "3" + "	" + "4");
  close(g);
}
EndTest();

// --- var[] array tests ---
// Note: doWrite takes ... var[] items, so the splat ...b must be the
// last unnamed argument.  To add endl, include it in the var[] array.

StartTest("var[] iteration: int, real, string, int");
{
  var[] b = new var[] {1, 2.5, "s", 7};
  file f = output(tmpfile);
  for (var s : b) write(f, s, endl);
  close(f);
  file g = input(tmpfile);
  string s;
  s = g; assert(s == "1");
  s = g; assert(s == "2.5");
  s = g; assert(s == "s");
  s = g; assert(s == "7");
  close(g);
}
EndTest();

StartTest("var[] splat: int, real, string, int");
{
  var[] b = new var[] {1, 2.5, "s", 7, endl};
  doWrite(...b);
  assert(doRead() == "1" + "	" + "2.5" + "	" + "s" + "	" + "7");
}
EndTest();

StartTest("var[] splat with label prefix");
{
  var[] b = new var[] {"pre:", 1, 2.5, "s", endl};
  doWrite(...b);
  assert(doRead() == "pre:1" + "	" + "2.5" + "	" + "s");
}
EndTest();

StartTest("var[] with bool, pair, triple, string");
{
  var[] c = new var[] {true, (1,2), (1,2,3), "end", endl};
  doWrite(...c);
  assert(doRead() == "true " + "	" + "(1,2)" + "	" + "(1,2,3)" + "	" + "end");
}
EndTest();

StartTest("var[] with Label (struct write method)");
{
  Label L = "hi";
  var[] v = new var[] {L, 42, endl};
  doWrite(...v);
  assert(doRead() == "\"hi\"" + "	" + "42");
}
EndTest();

StartTest("var[] iteration with Label");
{
  Label L = "world";
  var[] v = new var[] {L, 99};
  file f = output(tmpfile);
  for (var s : v) write(f, s, endl);
  close(f);
  file g = input(tmpfile);
  string s;
  s = g; assert(s == "\"world\"");
  s = g; assert(s == "99");
  close(g);
}
EndTest();

StartTest("var[] empty array");
{
  var[] e = new var[] {endl};
  doWrite(...e);
  assert(doRead() == "");
}
EndTest();

StartTest("var[] single element");
{
  var[] s = new var[] {42, endl};
  doWrite(...s);
  assert(doRead() == "42");
}
EndTest();

StartTest("var[] mixed with explicit args");
{
  var[] b = new var[] {10, 1.5, "mid", 20, endl};
  doWrite(...b);
  assert(doRead() == "10" + "	" + "1.5" + "	" + "mid" + "	" + "20");
}
EndTest();

StartTest("var[] length is correct");
{
  var[] b = new var[] {1, 2.5, "s", 7, true};
  assert(b.length == 5);
}
EndTest();

StartTest("var[] indexing");
{
  var[] b = new var[] {10, 20, 30};
  var x = b[0];
  var y = b[1];
  var z = b[2];
  var[] out = new var[] {x, y, z, endl};
  doWrite(...out);
  assert(doRead() == "10" + "	" + "20" + "	" + "30");
}
EndTest();

StartTest("var[] construction and length");
{
  // A var[] accepts an initializer list and reports its element count.
  // The adder methods (push/insert/append) are not available on var[]
  // because they would introduce raw untagged values; those errors are
  // asserted in errortests/var.asy.  This test only checks that the
  // array is constructed correctly.
  var[] b = new var[] {1, 2, 3};
  assert(b.length == 3);
}
EndTest();

StartTest("write(var[]) splats the array");
{
  file f = output(tmpfile);
  var[] b = new var[] {1, 2.5, "s", endl};
  write(f, b);
  close(f);
  assert(doRead() == "1" + '\t' + "2.5" + '\t' + "s");
}
EndTest();

StartTest("user-defined suffix is called");
{
  void bang(file f) { write(f, "!", endl); }
  file f = output(tmpfile);
  write(f, 1, "a", bang);
  close(f);
  assert(doRead() == "1" + '\t' + "a!");
}
EndTest();

StartTest("array values are written");
{
  file f = output(tmpfile);
  write(f, 1, new int[] {10,20}, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1");
  s = g; assert(s == "10");
  s = g; assert(s == "20");
  close(g);
}
EndTest();

StartTest("2-D array is written in mixed write");
{
  file f = output(tmpfile);
  write(f, new int[][] {{1,2},{3,4}}, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1" + '\t' + "2");
  s = g; assert(s == "3" + '\t' + "4");
  close(g);
}
EndTest();

StartTest("3-D array is written in mixed write");
{
  file f = output(tmpfile);
  write(f, new int[][][] {{{1,2},{3,4}},{{5,6},{7,8}}}, endl);
  close(f);
  file g = input(tmpfile);
  string s;
  s = g; assert(s == "1" + '\t' + "2");
  s = g; assert(s == "3" + '\t' + "4");
  s = g; assert(s == "");
  s = g; assert(s == "5" + '\t' + "6");
  s = g; assert(s == "7" + '\t' + "8");
  close(g);
}
EndTest();

StartTest("existing type-specific overloads still work");
{
  int x = 42;
  real y = 3.14;
  string s = "hello";
  pair p = (1,2);
  doWrite(x, y, endl);
  assert(doRead() == "42" + "	" + "3.14");
  // string+int matches type-specific write(file,string,int,suffix)
  // which treats the string as a label (no tab before int).
  doWrite(s, x, endl);
  assert(doRead() == "hello" + "42");
  doWrite(p, endl);
  assert(doRead() == "(1,2)");
}
EndTest();

// --- var-typed value regression tests ---
// These feed raw values into var slots (a var parameter / a var value read
// back out) and then pass them to the heterogeneous write().  Before the
// tagged_var invariant was enforced at the var-write boundary (exp.cc), a var
// parameter fed a literal produced a tagged_var with a var tag but a raw
// value, and write_var dereferenced that raw value as a tagged_var* pointer,
// segfaulting.

StartTest("var parameter fed an int literal");
{
  writeVarValue(5);
  assert(doRead() == "5");
}
EndTest();

StartTest("var parameter fed real, string, pair, bool");
{
  writeVarValue(2.5);
  assert(doRead() == "2.5");
  writeVarValue("hi");
  assert(doRead() == "hi");
  writeVarValue((1,2));
  assert(doRead() == "(1,2)");
  writeVarValue(true);
  assert(doRead() == "true ");
}
EndTest();

StartTest("var parameter fed a var[] element");
{
  var[] b = new var[] {7, 3.5, "s"};
  writeVarValue(b[0]);
  assert(doRead() == "7");
  writeVarValue(b[1]);
  assert(doRead() == "3.5");
  writeVarValue(b[2]);
  assert(doRead() == "s");
}
EndTest();

StartTest("var value read from var[] and reassigned stays writeable");
{
  var[] b = new var[] {41, 1.5};
  var x = b[0];
  var y = x;
  writeVarValue(y);
  assert(doRead() == "41");
  var z = b[1];
  writeVarValue(z);
  assert(doRead() == "1.5");
}
EndTest();
