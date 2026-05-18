// Template import errors.
{
  // Need to specify new name.
  access somefilename(T=int);
  // "as" misspelled
  access somefilename(T=int) notas somefilename_int;
  // missing keyword
  access somefilename(int) as somefilename_int;
  // Templated import unsupported
  import somefilename(T=int);
  // unexpected template parameters
  access errortestNonTemplate(T=int) as version;
}
{
  typedef import(T);  // this file isn't accessed as a template
  import typedef(T);  // should be "typedef import"
}
{
  // wrong number of parameters
  access errortestBrokenTemplate(A=int, B=string) as ett_a;
  // third param incorrectly named
  access errortestBrokenTemplate(A=int, B=string, T=real) as ett_b;
  // keywords in wrong order
  access errortestBrokenTemplate(A=int, C=real, B=string) as ett_c;
  // errortestBrokenTemplate.asy has extra "typedef import"
  access errortestBrokenTemplate(A=int, B=string, C=real) as ett_d;
  // expected template parameters
  access errortestBrokenTemplate as ett_e;
}
{
  // Non-statically nested types cannot be used as template parameters.
  struct A {
    struct B {
      autounravel int x;
    }
    access somefilename(T=B) as somefilename_B;
  }
  A a;
  access somefilename(T=a.B) as somefilename_B;
  access somefilename(T=A.B) as somefilename_B;
}
{
  // no error
  access errortestTemplate(A=int, B=string) as eft;
  // wrongly ordered names after correct load
  access errortestTemplate(B=int, A=string) as eft;
  // completely wrong names after correct load
  access errortestTemplate(C=int, D=string) as eft;
  // first name correct, second name wrong
  access errortestTemplate(A=int, D=string) as eft;
  // first name wrong, second name correct
  access errortestTemplate(C=int, B=string) as eft;
  // too few parameters
  access errortestTemplate(A=int) as eft;
  // too many parameters
  access errortestTemplate(A=int, B=string, C=real) as eft;
  // templated imports cannot be run directly
  include errortestTemplate;
}
