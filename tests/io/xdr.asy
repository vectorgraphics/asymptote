import TestLib;

void test(bool r, bool i) {
  file fout=output("xdata",mode="xdr").word().singlereal(r).singleint(i);
  write(fout,"test");
  write(fout,1);
  write(fout,2.0);
  write(fout,(3,4));
  write(fout,(5,6,7));
  close(fout);

  file fin=input("xdata",mode="xdr").word().singlereal(r).singleint(i);
  string a=fin;
  assert(a == "test");
  int b=fin;
  assert(b == 1);
  real c=fin;
  assert(c == 2.0);
  pair d=fin;
  assert(d == (3,4));
  triple e=fin;
  assert(e == (5,6,7));
}

StartTest("xdr: single real, single int");
{
  test(true,true);
}
EndTest();

StartTest("xdr: single real, double int");
{
  test(true,false);
}
EndTest();

StartTest("xdr: double real, single int");
{
  test(false,true);
}
EndTest();

StartTest("xdr: double real, double int");
{
  test(false,false);
}
EndTest();

StartTest("xdr: character");
{
  file fout=output("xdata",mode="xdr");
  write(fout,"a");
  write(fout,"b");
  close(fout);

  file fin=input("xdata",mode="xdr");
  string a=getc(fin);
  assert(a == "a");
  string b=getc(fin);
  assert(b == "b");
}
EndTest();
