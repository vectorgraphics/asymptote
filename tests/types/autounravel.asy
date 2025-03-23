import TestLib;

StartTest('autounravel: struct declaration');
{
  struct A {
    autounravel int x = -1;
  }
  assert(x == -1);
  assert(A.x == -1);
  x = -2;
  assert(A.x == -2);
  A.x = -3;
  assert(x == -3);
}
EndTest();

StartTest('autounravel: typedef');
{
  struct B {
    static struct C {
      autounravel int y = -1;
    }
  }
  typedef B.C C;
  assert(y == -1);
  assert(C.y == -1);
  y = -2;
  assert(C.y == -2);
  C.y = -3;
  assert(y == -3);
}
EndTest();

StartTest('autounravel: unravel *');
{
  struct B {
    static struct C {
      autounravel int y = -1;
    }
  }
  unravel B;
  assert(y == -1);
  assert(C.y == -1);
  y = -2;
  assert(C.y == -2);
  C.y = -3;
  assert(y == -3);
}
EndTest();

StartTest('autounravel: unravel');
{
  struct B {
    static struct C {
      autounravel int y = -1;
    }
  }
  from B unravel C;
  assert(y == -1);
  assert(C.y == -1);
  y = -2;
  assert(C.y == -2);
  C.y = -3;
  assert(y == -3);
}
EndTest();

StartTest('autounravel: field is unraveled');
{
  struct A {
    static int z = -1;
  }
  struct B {
    static struct C {
      autounravel from A unravel z as zz;
    }
  }
  from B unravel C;
  assert(zz == -1);
  assert(C.zz == -1);
  zz = -2;
  assert(C.zz == -2);
  assert(A.z == -2);
  C.zz = -3;
  assert(zz == -3);
  assert(A.z == -3);
  A.z = -4;
  assert(C.zz == -4);
  assert(zz == -4);
}
EndTest();

StartTest('autounravel: whole struct is unraveled');
{
  struct A {
    static int z = -1;
  }
  struct B {
    static struct C {
      autounravel unravel A;
    }
  }
  from B unravel C;
  assert(z == -1);
  assert(C.z == -1);
  z = -2;
  assert(C.z == -2);
  assert(A.z == -2);
  C.z = -3;
  assert(z == -3);
  assert(A.z == -3);
  A.z = -4;
  assert(C.z == -4);
  assert(z == -4);
}
EndTest();

StartTest('autounravel: function');
{
  struct B {
    static struct C {
      autounravel int y() { return -1; }
    }
  }
  from B unravel C;
  assert(y() == -1);
  assert(C.y() == -1);
  y = new int() { return -2; };
  assert(C.y() == -2);
  assert(y() == -2);
  C.y = new int() { return -3; };
  assert(y() == -3);
  assert(C.y() == -3);
}
EndTest();

StartTest('autounravel: multiple fields');
{
  struct B {
    static struct C {
      autounravel int y = -1;
      autounravel int z = 1;
    }
  }
  from B unravel C;
  assert(y == -1);
  assert(C.y == -1);
  assert(z == 1);
  assert(C.z == 1);
  y = -2;
  assert(C.y == -2);
  assert(z == 1);
  assert(C.z == 1);
  C.z = 2;
  assert(y == -2);
  assert(C.y == -2);
  assert(z == 2);
  assert(C.z == 2);
}
EndTest();

StartTest('autounravel: builtin operators');
{
  struct B {
    static struct C {}
  }
  from B unravel C;
  C x = new C;
  assert(alias(x, x));
  assert(!alias(x, null));
}
EndTest();