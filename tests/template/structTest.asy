struct BareStruct {
  static string testName = "bare struct";
}
struct A {
  static int global = 17;
  int local = 3;
}
access 'template/imports/structTemplate'(T=A, Lib=BareStruct) as bareStruct;

struct NestedStruct {
  static string testName = "nested struct";
}
struct B {
  static struct C {
    static int global = 17;
    int local = 3;
  }
}
access 'template/imports/structTemplate'(T=B.C, Lib=NestedStruct)
    as nestedStruct;

struct InnerStruct {
  static string testName = "inner struct";
}
struct D {
  struct E {
    static int global = 17;
    int local = 3;
  }
}
D d;
access 'template/imports/structTemplate'(T=d.E, Lib=InnerStruct)
    as innerStruct;

struct ImportedStruct {
  static string testName = "imported struct";
}
access 'template/imports/notTemplate' as notTemplate;
access 'template/imports/structTemplate'(T=notTemplate.A, Lib=ImportedStruct)
    as importedStruct;