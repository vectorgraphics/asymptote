import TestLib;
StartTest("constructor");

{ // Basic usage. {{{1
  struct Foo {
    int x;
    int y;

    void operator init() {
      x=2; y=3;
    }

    void operator init(int x, int y=2x) {
      this.x=x;  this.y=y;
    }

    void operator init(string s ... int[] vals) {
      for (int i=0; i<vals.length; ++i)
        x+=vals[i];
      y=vals.length;
    }

    static {
      {
        Foo f=new Foo;
        assert(f.x==0);
        assert(f.y==0);
      }
      {
        Foo f=Foo();
        assert(f.x==2);
        assert(f.y==3);
      }
      {
        Foo f=Foo(7);
        assert(f.x==7);
        assert(f.y==14);
      }
      {
        Foo f=Foo(5,40);
        assert(f.x==5);
        assert(f.y==40);
      }
      {
        Foo f=Foo(y=5,x=40);
        assert(f.x==40);
        assert(f.y==5);
      }
      {
        Foo f=Foo("hello", 1,2,3,4);
        assert(f.x==10);
        assert(f.y==4);
      }
    }
  }

  {
    Foo f=new Foo;
    assert(f.x==0);
    assert(f.y==0);
  }
  {
    Foo f=Foo();
    assert(f.x==2);
    assert(f.y==3);
  }
  {
    Foo f=Foo(7);
    assert(f.x==7);
    assert(f.y==14);
  }
  {
    Foo f=Foo(5,40);
    assert(f.x==5);
    assert(f.y==40);
  }
  {
    Foo f=Foo(y=5,x=40);
    assert(f.x==40);
    assert(f.y==5);
  }
  {
    Foo f=Foo("hello", 1,2,3,4);
    assert(f.x==10);
    assert(f.y==4);
  }
}

{ // Assignment {{{1
  struct Foo {
    int x;

    static int Foo() {
      return 7;
    }

    static assert(Foo()==7);

    void five() {
      x=5;
    }

    void six(string s) {
      x=6;
    }

    void operator init()=five;

    void operator init(string s) {
      assert(false);
    }
    operator init=six;

    static {
      assert(Foo().x==5);
      assert(Foo("milk").x==6);
    }

    void operator init(int x) {
      this.x=2x;
    }

    static Foo constructor(int)=Foo;

    static {
      assert(constructor(5).x==10);
    }
  }

  assert(Foo().x==5);
  assert(Foo("milk").x==6);
  assert(Foo.constructor(5).x==10);
}

{ // Permission {{{1
  int Foo() {
    return 3;
  }

  struct Foo {
    int x;

    private void operator init() {
      x=2;
    }
  }
  // The implicit constructor should not be unravelled, as it is private.
  assert(Foo()==3);
}
{ // Shadowing {{{1
  struct Foo {
    static Foo Foo() {
     return null;
    }

    static assert(Foo()==null);

    void operator init() {
    }

    static assert(Foo()!=null);
  }

  assert(Foo()!=null);

  // Can't use Bar because of conflicts with the drawing command.
  struct Barr {
    void operator init() {
    }

    static assert(Barr()!=null);

    static Barr Barr() {
     return null;
    }

    static assert(Barr()==null);
  }

  assert(Barr()!=null);

  struct Blah {
    void operator init() {
    }

    static assert(Blah()!=null);

    static Blah Blah() {
     return null;
    }

    static assert(Blah()==null);
  }
  from Blah unravel Blah;

  assert(Blah()==null);
}

{ // Static {{{1
  struct Foo {
    static int Foo() {
      return 7;
    }

    static void operator init() {
    }

    static assert(Foo()==7);
  }

  struct Barr {
    int x;

    void operator init() {
      x=77;
    }

    static assert(Barr().x==77);

    static void operator init() {
      assert(false);
    }

    static assert(Barr().x==77);
  }

  assert(Barr().x==77);
}

{ // Struct name shadowing. {{{1
  struct Foo {
    int x;
    typedef int Foo;

    void operator init() {
      x=89;
    }

    static assert(Foo().x==89);
  }

  assert(Foo().x==89);

  struct Barr {
    int x;
    struct Barr {
    }

    void operator init() {
      x=89;
    }

    static assert(Barr().x==89);
  }

  assert(Barr().x==89);
}
{ // Function Comparison
  struct A {
    // Defines a function A A(int).
    void operator init(int x) {}
  }
  assert(!(A == null));
  assert(A != null);
}

// File-level {{{1
// This assumes the file is named constructor.asy

int constructor() {
  return 44;
}
void operator init() {
  assert(false);
}
assert(constructor()==44);

EndTest(); // {{{1

