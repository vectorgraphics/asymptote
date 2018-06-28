// Asymptote module implementing rational arithmetic.

int gcd(int m, int n)
{
  if(m < n) {
    int temp=m;
    m=n;
    n=temp;
  }
  while(n != 0) {
    int r=m % n;
    m=n;
    n=r;
  }       
  return m;
}

int lcm(int m, int n)
{
  return m#gcd(m,n)*n;
}

struct rational {
  int p=0,q=1;
  void reduce() {
    int d=gcd(p,q);
    if(abs(d) > 1) {
      p #= d;
      q #= d;
    }
    if(q <= 0) {
      if(q == 0) abort("Division by zero");
      p=-p;
      q=-q;
    }
  }
  void operator init(int p=0, int q=1, bool reduce=true) {
    this.p=p;
    this.q=q;
    if(reduce) reduce();
  }
}

rational operator cast(int p) {
  return rational(p,false);
}

rational[] operator cast(int[] a) {
  return sequence(new rational(int i) {return a[i];},a.length);
}

rational[][] operator cast(int[][] a) {
  return sequence(new rational[](int i) {return a[i];},a.length);
}

real operator ecast(rational r) {
  return r.p/r.q;
}

rational operator -(rational r)
{
  return rational(-r.p,r.q,false);
}

rational operator +(rational r, rational s)
{
  return rational(r.p*s.q+s.p*r.q,r.q*s.q);
}

rational operator -(rational r, rational s)
{
  return rational(r.p*s.q-s.p*r.q,r.q*s.q);
}

rational operator *(rational r, rational s)
{
  return rational(r.p*s.p,r.q*s.q);
}

rational operator /(rational r, rational s)
{
  return rational(r.p*s.q,r.q*s.p);
}

bool operator ==(rational r, rational s)
{
  return r.p == s.p && r.q == s.q;
}

bool operator !=(rational r, rational s)
{
  return r.p != s.p || r.q != s.q;
}

bool operator <(rational r, rational s)
{
  return r.p*s.q-s.p*r.q < 0;
}

bool operator >(rational r, rational s)
{
  return r.p*s.q-s.p*r.q > 0;
}

bool operator <=(rational r, rational s)
{
  return r.p*s.q-s.p*r.q <= 0;
}

bool operator >=(rational r, rational s)
{
  return r.p*s.q-s.p*r.q >= 0;
}

bool[] operator ==(rational[] r, rational s)
{
  return sequence(new bool(int i) {return r[i] == s;},r.length);
}

bool operator ==(rational[] r, rational[] s)
{
  if(r.length != s.length)
    abort(" operation attempted on arrays of different lengths: "+
          string(r.length)+" != "+string(s.length));
  return all(sequence(new bool(int i) {return r[i] == s[i];},r.length));
}

bool operator ==(rational[][] r, rational[][] s)
{
  if(r.length != s.length)
    abort(" operation attempted on arrays of different lengths: "+
          string(r.length)+" != "+string(s.length));
  return all(sequence(new bool(int i) {return r[i] == s[i];},r.length));
}

rational min(rational a, rational b)
{
  return a <= b ? a : b;
}

rational max(rational a, rational b)
{
  return a >= b ? a : b;
}


void write(string s="", rational r, suffix suffix=endl) {
  if(r.q == 1)
    write(s+string(r.p),suffix);
  else
    write(s+string(r.p)+"/"+string(r.q),suffix);
}

void write(string s="", rational[] a, suffix suffix=endl) {
  if(s != "")
    write(s);
  for(int i=0; i < a.length; ++i) {
    write(i,none);
    write(':\t',a[i]);
  }
  write(suffix);
}

void write(string s="", rational[][] a, suffix suffix=endl) {
  if(s != "")
    write(s);
  for(int i=0; i < a.length; ++i) {
    rational[] ai=a[i];
    for(int j=0; j < ai.length; ++j) {
      write(ai[j],tab);
    }
    write();
  }
  write(suffix);
}

bool rectangular(rational[][] m)
{
  int n=m.length;
  if(n > 0) {
    int m0=m[0].length;
    for(int i=1; i < n; ++i)
      if(m[i].length != m0) return false;
  }
  return true;
}

rational sum(rational[] a)
{
  rational sum;
  for(rational r:a)
    sum += r;
  return sum;
}

rational max(rational[] a)
{
  rational M=a[0];
  for(rational r:a)
    M=max(M,r);
  return M;
}

rational abs(rational r)
{
  return rational(abs(r.p),r.q,false);
}


/*
rational r=rational(1,3)+rational(1,4);
write(r == rational(1,12));
write(r);
real x=r;
write(x);

rational r=3;
write(r);

write(gcd(-8,12));
write(rational(-36,-14));

int[][] a=new int[][] {{1,2},{3,4}};
rational[][] r=a;
write(r);

*/

