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

// Test the curlSpecifier type.
{
  curlSpecifier operator ..(curlSpecifier spec, string b) {
    return spec;
  }
  curlSpecifier operator ..(string a, curlSpecifier spec) {
    return spec;
  }
  curlSpecifier operator ..(string a, curlSpecifier spec, string b) {
    return spec;
  }

  curlSpecifier spec="hello"{curl 3}.."joe";
  assert(spec.value==3);
  assert(spec.side==JOIN_OUT);

  curlSpecifier spec="hello"..{curl 7}"joe";
  assert(spec.value==7);
  assert(spec.side==JOIN_IN);
}

EndTest();

