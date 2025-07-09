import TestLib;

void test(bool r, bool i, string mode) {
  string name="xdata";
  string s="testing";
  file fout=output(name,mode=mode).word().singlereal(r).singleint(i);
  write(fout,s);
  write(fout,1);
  fout.word(false);
  write(fout,"a");
  write(fout,"b");
  fout.word(true);
  write(fout,2.0);
  write(fout,(3,4));
  write(fout,(5,6,7));
  close(fout);

  file fin=input(name,mode=mode).word().singlereal(r).singleint(i);
  string a=fin;
  assert(a == s);
  int b=fin;
  assert(b == 1);
  fin.word(false);
  assert(getc(fin) == "a");
  assert(getc(fin) == "b");
  fin.word(true);
  real c=fin;
  assert(c == 2.0);
  pair d=fin;
  assert(d == (3,4));
  triple e=fin;
  assert(e == (5,6,7));
}

string text[]={"double","single"};
string mode[]={"xdr","xdrgz"};

for(int z=0; z <= 1; ++z) {
  for(int r=0; r <= 1; ++r) {
    for(int i=0; i <= 1; ++i) {
      StartTest(mode[z]+": "+text[r]+" real, "+text[i]+" int");
      test(r == 1,i == 1,mode[z]);
      EndTest();
    }
  }
}
