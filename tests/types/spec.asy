import TestLib;
StartTest("spec");

// Test if the cycle keyword can be used in different contexts.
{
  int operator cast(cycleToken) {
    return 55;
  }
  int x=cycle;
  assert(x==55);
}

// Test the tensionSpecifier type.
{
  tensionSpecifier operator ..(string a, tensionSpecifier t, string b) {
    return t;
  }
  tensionSpecifier t="hello" .. tension 2 .. "joe";
  assert(t.out==2);
  assert(t.in==2);
  assert(t.atLeast==false);

  tensionSpecifier t="hello" .. tension 3 and 2 .. "joe";
  assert(t.out==3);
  assert(t.in==2);
  assert(t.atLeast==false);

  tensionSpecifier t="hello" .. tension atleast 7 .. "joe";
  assert(t.out==7);
  assert(t.in==7);
  assert(t.atLeast==true);

  tensionSpecifier t="hello" .. tension atleast 3 and 2 .. "joe";
  assert(t.out==3);
  assert(t.in==2);
  assert(t.atLeast==true);
}
EndTest();

