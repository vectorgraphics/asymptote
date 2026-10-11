// Heterogeneous write() tests.
import TestLib;

string tmpfile = "write_test_tmp.txt";

string doRead()
{
  file g = input(tmpfile);
  string s = g;
  close(g);
  return s;
}

StartTest("write single int");
{
  file f = output(tmpfile);
  write(f, 42, endl);
  close(f);
  assert(doRead() == "42");
}
EndTest();

StartTest("write single real");
{
  file f = output(tmpfile);
  write(f, 3.14, endl);
  close(f);
  assert(doRead() == "3.14");
}
EndTest();

StartTest("write single string");
{
  file f = output(tmpfile);
  write(f, "hello", endl);
  close(f);
  assert(doRead() == "hello");
}
EndTest();

StartTest("write two ints");
{
  file f = output(tmpfile);
  write(f, 42, 7, endl);
  close(f);
  assert(doRead() == "42" + '\t' + "7");
}
EndTest();

StartTest("write int and real");
{
  file f = output(tmpfile);
  write(f, 42, 3.14, endl);
  close(f);
  assert(doRead() == "42" + '\t' + "3.14");
}
EndTest();

StartTest("write real and pair");
{
  file f = output(tmpfile);
  write(f, 1.5, (1,2), endl);
  close(f);
  assert(doRead() == "1.5" + '\t' + "(1,2)");
}
EndTest();

StartTest("write three heterogeneous");
{
  file f = output(tmpfile);
  write(f, 42, 3.14, "end", endl);
  close(f);
  assert(doRead() == "42" + '\t' + "3.14" + '\t' + "end");
}
EndTest();

StartTest("write label and int");
{
  file f = output(tmpfile);
  write(f, "x =", 42, endl);
  close(f);
  assert(doRead() == "x =42");
}
EndTest();

StartTest("write label and two values");
{
  file f = output(tmpfile);
  write(f, "x =", 42, 7, endl);
  close(f);
  assert(doRead() == "x =42" + '\t' + "7");
}
EndTest();

StartTest("write two strings");
{
  file f = output(tmpfile);
  write(f, "a", "b", endl);
  close(f);
  assert(doRead() == "ab");
}
EndTest();

StartTest("write two pairs");
{
  file f = output(tmpfile);
  write(f, (1,2), (3,4), endl);
  close(f);
  assert(doRead() == "(1,2)" + '\t' + "(3,4)");
}
EndTest();

StartTest("write two bools");
{
  file f = output(tmpfile);
  write(f, true, false, endl);
  close(f);
  assert(doRead() == "true " + '\t' + "false ");
}
EndTest();

StartTest("write Label (struct method)");
{
  Label L = "hi";
  file f = output(tmpfile);
  write(f, L, 42, endl);
  close(f);
  assert(doRead() == "\"hi\"" + '\t' + "42");
}
EndTest();

StartTest("write two labels (struct method)");
{
  Label A = "first";
  Label B = "second";
  file f = output(tmpfile);
  write(f, A, B, endl);
  close(f);
  assert(doRead() == "\"first\"" + '\t' + "\"second\"");
}
EndTest();

StartTest("write string prefix and label");
{
  Label L = "value";
  file f = output(tmpfile);
  write(f, "key", L, endl);
  close(f);
  assert(doRead() == "key\"value\"");
}
EndTest();

StartTest("write array of Labels (struct method)");
{
  Label[] L = {Label("a"), Label("b")};
  file f = output(tmpfile);
  write(f, 1, L, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1");
  s = g; assert(s == "\"a\"");
  s = g; assert(s == "\"b\"");
  close(g);
}
EndTest();

StartTest("write with flush suffix (no newline)");
{
  file f = output(tmpfile);
  write(f, 42, flush);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "42");
  close(g);
}
EndTest();

StartTest("write with endl suffix (newline)");
{
  file f = output(tmpfile);
  write(f, 42, endl);
  close(f);
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
  string s; s = g; assert(s == "1" + '\t' + "2");
  s = g; assert(s == "3" + '\t' + "4");
  close(g);
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
  file f = output(tmpfile);
  write(f, x, y, endl);
  close(f);
  assert(doRead() == "42" + '\t' + "3.14");
  f = output(tmpfile);
  write(f, s, x, endl);
  close(f);
  assert(doRead() == "hello" + "42");
  f = output(tmpfile);
  write(f, p, endl);
  close(f);
  assert(doRead() == "(1,2)");
}
EndTest();

StartTest("write path");
{
  path p = unitcircle;
  file f = output(tmpfile);
  write(f, p, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g;
  assert(s == "(1,0).. controls (1,0.5522847) and (0.5522847,1)");
  s = g; assert(s == " ..(0,1).. controls (-0.5522847,1) and (-1,0.5522847)");
  s = g; assert(s == " ..(-1,0).. controls (-1,-0.5522847) and (-0.5522847,-1)");
  s = g; assert(s == " ..(0,-1).. controls (0.5522847,-1) and (1,-0.5522847)");
  s = g; assert(s == " ..cycle");
  close(g);
}
EndTest();

StartTest("write guide");
{
  guide g2 = (0,0)..(1,1)..(2,0);
  file f = output(tmpfile);
  write(f, g2, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g;
  assert(s == "(0,0)");
  s = g; assert(s == "..(1,1)");
  s = g; assert(s == "..(2,0)");
  close(g);
}
EndTest();

StartTest("write path and guide together");
{
  path p = (0,0)--(1,0)--cycle;
  guide g2 = (0,0)..(1,1);
  file f = output(tmpfile);
  write(f, p, g2, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g;
  assert(find(s, "--") >= 0);
  s = g; assert(find(s, "..") >= 0);
  close(g);
}
EndTest();

StartTest("write file only (no data) is a no-op");
{
  file f = output(tmpfile);
  write(f);
  close(f);
  assert(doRead() == "");
}
EndTest();

StartTest("mixed scalars and arrays");
{
  file f = output(tmpfile);
  write(f, 1, new int[] {2, 3}, 4.5, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1");
  s = g; assert(s == "2");
  s = g; assert(s == "3" + '\t' + "4.5");
  close(g);
}
EndTest();

StartTest("named arguments");
{
  file f = output(tmpfile);
  write(file=f, 1, 2.5, suffix=endl);
  write(f, s="x=", 3, endl);
  write(suffix=endl, 4, s="y=", "z", file=f);
  write(f, "a", s="b", endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1" + '\t' + "2.5");
  s = g; assert(s == "x=3");
  s = g; assert(s == "y=4" + '\t' + "z");
  s = g; assert(s == "ba");
  close(g);
}
EndTest();

StartTest("rest argument");
{
  file f = output(tmpfile);
  write(f, 1, 2, endl ... new int[] {3, 4});
  write(f, "x=", 1.5 ... new real[] {});
  write(f, endl ... new string[] {"a", "b"});
  write(f, "c", endl ... new string[] {"a", "b"});
  write(f, 0, endl ... new Label[] {Label("L")});
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1" + '\t' + "2" + '\t' + "3" + '\t' + "4");
  s = g; assert(s == "x=1.5a" + '\t' + "b");
  s = g; assert(s == "ca" + '\t' + "b");
  s = g; assert(s == "0" + '\t' + "\"L\"");
  close(g);
}
EndTest();

StartTest("overloaded name is written as its value");
{
  int h = 3;
  int h(int x) { return x; }
  file f = output(tmpfile);
  write(f, h, 2.5, endl);
  close(f);
  assert(doRead() == "3" + '\t' + "2.5");
}
EndTest();

StartTest("struct without a writer is written field by field");
{
  struct Inner { int a = 2; string s = "hi"; }
  struct Outer {
    int x = 1;
    private real y = 2.5;
    Inner inner;
    int[] list = {1, 2, 3};
    pair z = (1, 2);
    Outer next;
    void method() { }
    static int shared = 7;
  }
  Outer o;
  file f = output(tmpfile);
  write(f, o, endl);
  write(f, "o=", 5, o, endl);
  write(f, new Outer[] {o, null}, endl);
  struct Empty { }
  write(f, new Empty, endl);
  close(f);
  string expected = "(x=1, y=2.5, inner=(a=2, s=\"hi\"), list={1, 2, 3}, " +
                    "z=(1,2), next=null)";
  file g = input(tmpfile);
  string s; s = g; assert(s == expected);
  s = g; assert(s == "o=5" + '\t' + expected);
  s = g; assert(s == expected);
  s = g; assert(s == "null");
  s = g; assert(s == "()");
  close(g);
}
EndTest();

StartTest("struct fields are shown to a limited depth");
{
  struct Node { int v; Node next; }
  Node chain(int n) {
    Node node = new Node;
    node.v = n;
    if (n > 1) node.next = chain(n - 1);
    return node;
  }
  int depth = settings.structdepth;
  file f = output(tmpfile);
  write(f, chain(5), endl);
  settings.structdepth = 1;
  write(f, chain(5), endl);
  settings.structdepth = depth;
  close(f);
  file g = input(tmpfile);
  string s; s = g;
  assert(s == "(v=5, next=(v=4, next=(v=3, next=<Node>)))");
  s = g; assert(s == "(v=5, next=<Node>)");
  close(g);
}
EndTest();

StartTest("struct that contains itself is written once");
{
  struct Loop { int v = 1; Loop self; Loop[] list; Loop other; }
  Loop a, b;
  a.self = a;
  a.list.push(a);
  a.other = b;
  b.other = a;
  int depth = settings.structdepth;
  int limit = settings.structlimit;
  settings.structdepth = 1000000;
  settings.structlimit = 0;
  file f = output(tmpfile);
  write(f, a, endl);
  settings.structdepth = depth;
  settings.structlimit = limit;
  close(f);
  assert(doRead() == "(v=1, self=<Loop, cyclic>, list={<Loop, cyclic>}, " +
         "other=(v=1, self=null, list={}, other=<Loop, cyclic>))");
}
EndTest();

StartTest("long struct output is cut short with matching brackets");
{
  struct Data { int[][] m = {{1, 2}, {3}}; string text = "abcdefghijklmnop"; }
  int limit = settings.structlimit;
  file f = output(tmpfile);
  settings.structlimit = 30;
  write(f, new Data, endl);
  settings.structlimit = 8;
  write(f, new Data, endl);
  settings.structlimit = 0;
  write(f, new Data, endl);
  settings.structlimit = limit;
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "(m={{1, 2}, {3}}, text=\"abcdef...\")");
  s = g; assert(s == "(m={{1, ...}})");
  s = g; assert(s == "(m={{1, 2}, {3}}, text=\"abcdefghijklmnop\")");
  close(g);
}
EndTest();

StartTest("struct with its own writer is written by it at any depth");
{
  struct Method {
    int v = 3;
    void write(file f, suffix s) { write(f, "M", v); s(f); }
  }
  struct Static {
    int v = 4;
    autounravel void write(file f, Static x, suffix s) {
      write(f, "S", x.v);
      s(f);
    }
  }
  struct Private {
    int v = 5;
    private void write(file f, suffix s) { write(f, "hidden"); }
  }
  struct Holder { Method m; Static[] list = {new Static}; Private p; Label L = "lab"; }
  file f = output(tmpfile);
  write(f, new Method, new Static, new Private, endl);
  write(f, new Holder, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "M3" + '\t' + "S4" + '\t' + "(v=5)");
  s = g; assert(s == "(m=M3, list={S4}, p=(v=5), L=\"lab\")");
  close(g);
}
EndTest();

StartTest("functions and other values with no text are named by type");
{
  using Func = real(real);
  struct Table {
    Func f = sin;
    Func[] fs = {sin, cos};
    string name = "t";
    real method(real x) { return x; }
  }
  real twice(real x) { return 2x; }
  file f = output(tmpfile);
  write(f, 1, twice, 2, endl);
  write(f, new Func[] {sin, cos}, endl);
  write(f, 0, stdout, endl);
  write(f, new Table, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "1" + '\t' + "<real(real x)>" + '\t' + "2");
  s = g; assert(s == "<real(real)[] of length 2>");
  s = g; assert(s == "0" + '\t' + "<file>");
  s = g; assert(s == "(f=<real(real)>, fs=<real(real)[] of length 2>, " +
                     "name=\"t\")");
  close(g);
}
EndTest();

StartTest("bool3 is written by its own writer");
{
  bool3 b;
  struct Flags { bool3 a; bool3 b = true; }
  file f = output(tmpfile);
  write(f, b, endl);
  write(f, "b=", b, endl);
  write(f, new Flags, endl);
  close(f);
  file g = input(tmpfile);
  string s; s = g; assert(s == "default");
  s = g; assert(s == "b=default");
  s = g; assert(s == "(a=default, b=true)");
  close(g);
}
EndTest();
