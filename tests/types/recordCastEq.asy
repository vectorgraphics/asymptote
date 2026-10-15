
import TestLib;

StartTest("recordCastEq: record != string when string casts to record");
{
  // Struct with a cast FROM string TO Tag (like Label).
  // Only one direction: string -> Tag.  Tag cannot be cast to string.
  struct Tag {
    string s;
    void operator init(string s="") { this.s = s; }
  }
  Tag operator cast(string s) { Tag t; t.s = s; return t; }

  Tag t = Tag("hello");

  // record != string: no concrete !=(Tag, string) exists, and Tag cannot
  // be cast to string, so the open-builtin record != is used.  The string
  // operand is implicitly cast to Tag, then compared by identity.
  assert(t != "");        // fresh Tag("") != t  (different instances)
  assert(t != "hello");   // fresh Tag("hello") != t (different instances)

  // Same-instance comparison still works.
  Tag t2 = t;
  assert(t == t2);
  assert(!(t != t2));

  // null comparisons.
  assert(t != null);
  assert(!(t == null));
}
EndTest();

StartTest("recordCastEq: record == castable non-record, other direction");
{
  // real casts to N (but not the reverse), mirroring Label/string.
  struct N {
    real v;
    void operator init(real v=0) { this.v = v; }
  }
  N operator cast(real v) { N n; n.v = v; return n; }

  N a = new N; a.v = 1.0;
  N b = new N; b.v = 1.0;

  // real is cast to N, identity comparison.
  assert(a != 1.0);  // different instances
  assert(a != 2.0);

  // Same-type identity semantics unchanged.
  assert(a != b);    // different instances, same value
  assert(a == a);
}
EndTest();

StartTest("recordCastEq: record vs castable int");
{
  struct A { int x; }
  A operator cast(int x) { A a; a.x = x; return a; }

  A a = new A; a.x = 1;

  // Same-type identity semantics.
  A a2 = new A; a2.x = 1;
  assert(a != a2);  // different instances

  // int casts to A.
  assert(a != 1);   // different instances
}
EndTest();

