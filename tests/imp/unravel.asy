import TestLib;
StartTest("unravel");
{
  struct A {
    int x=1, y=2, z=3;
    int y() { return 7; }
  }

  A a=new A;
  unravel a;
  assert(x==1);
  assert(y==2);
  assert(z==3);
  assert(y()==7);
}
{
  struct A {
    private int x=1;
    int y=2, z=3;
    int y() { return 7; }
  } 

  int x=5;
  A a=new A;
  unravel a;
  assert(x==5);
  assert(y==2);
  assert(z==3);
}
{
  struct A {
    public int x=1;
    int y=2, z=3;
    int y() { return 7; }
  }

  int z=5;
  A a=new A;
  from a unravel x,y;
  assert(x==1);
  assert(y==2);
  assert(z==5);
  assert(y()==7);
}
{
  struct A {
    public int x=1;
    int y=2, z=3;
    int y() { return 7; }
  }

  int y=4;
  int z=5;
  A a=new A;
  from a unravel x,y as blah;
  assert(x==1);
  assert(y==4);
  assert(blah==2);
  assert(z==5);
  assert(blah()==7);
}
{
  struct A {
    struct B {
      static int x=4;
    }
  }
  A a=new A;
  int x=3;
  from a.B unravel x;
  assert(x==4);
}
{
  struct A {
    struct B {
      static int x=4;
    }
  }
  A a=new A;
  A.B b=new a.B;
  int x=3;
  from b unravel x;
  assert(x==4);
}
{
  struct A {
    static struct B {
      static int x=4;
    }
  }
  int x=3;
  from A.B unravel x;
  assert(x==4);
}
EndTest();
