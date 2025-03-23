struct BareStruct {
  static string testName = "bare struct";
}
struct A {
  static int global = 17;
  int local = 3;
  autounravel int au = -1;
}
access 'template/imports/structTemplate'(T=A, Lib=BareStruct) as bareStruct;

struct NestedStruct {
  static string testName = "nested struct";
}
struct B {
  static struct C {
    static int global = 17;
    int local = 3;
    autounravel int au = -1;
  }
}
access 'template/imports/structTemplate'(T=B.C, Lib=NestedStruct)
    as nestedStruct;

struct DeeplyNestedStruct {
  static string testName = "deeply nested struct";
}

struct G {
  static struct H {
    static struct I {
      static int global = 17;
      int local = 3;
      autounravel int au = -1;
    }
  }
}
G g;

access 'template/imports/structTemplate'(T=g.H.I, Lib=DeeplyNestedStruct)
    as deeplyNestedStruct;

struct ImportedStruct {
  static string testName = "imported struct";
}
access 'template/imports/notTemplate' as notTemplate;
access 'template/imports/structTemplate'(T=notTemplate.A, Lib=ImportedStruct)
    as importedStruct;

struct NestedImport {
  static string testName = "nested import";
}
access 'template/imports/notTemplate2' as notTemplate2;
access 'template/imports/structTemplate'(T=notTemplate2.b.A, Lib=NestedImport)
    as nestedImport;