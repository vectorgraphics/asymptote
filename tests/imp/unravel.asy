import TestLib;
StartTest("unravel");
{
  struct A {
    int x=1, y=2, z=3;
    int y() { return 7; }
  }

  A a=new A;
  unravel a;
  Assert(x==1);
  Assert(y==2);
  Assert(z==3);
  Assert(y()==7);
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
  Assert(x==5);
  Assert(y==2);
  Assert(z==3);
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
  Assert(x==1);
  Assert(y==2);
  Assert(z==5);
  Assert(y()==7);
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
  Assert(x==1);
  Assert(y==4);
  Assert(blah==2);
  Assert(z==5);
  Assert(blah()==7);
}
EndTest();
