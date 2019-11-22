restricted int undefined=(intMax % 2 == 1) ? intMax : intMax-1;

restricted real inches=72;
restricted real inch=inches;
restricted real cm=inches/2.54;
restricted real mm=0.1cm;
restricted real bp=1;      // A PostScript point.
restricted real pt=72.0/72.27; // A TeX pt; smaller than a PostScript bp.
restricted pair I=(0,1);

restricted pair right=(1,0);
restricted pair left=(-1,0);
restricted pair up=(0,1);
restricted pair down=(0,-1);

restricted pair E=(1,0);
restricted pair N=(0,1);
restricted pair W=(-1,0);
restricted pair S=(0,-1);

restricted pair NE=unit(N+E);
restricted pair NW=unit(N+W);
restricted pair SW=unit(S+W);
restricted pair SE=unit(S+E);

restricted pair ENE=unit(E+NE);
restricted pair NNE=unit(N+NE);
restricted pair NNW=unit(N+NW);
restricted pair WNW=unit(W+NW);
restricted pair WSW=unit(W+SW);
restricted pair SSW=unit(S+SW);
restricted pair SSE=unit(S+SE);
restricted pair ESE=unit(E+SE);
  
restricted real sqrtEpsilon=sqrt(realEpsilon);
restricted pair Align=sqrtEpsilon*NE; 
restricted int mantissaBits=ceil(-log(realEpsilon)/log(2))+1;

restricted transform identity;
restricted transform zeroTransform=(0,0,0,0,0,0);

int min(... int[] a) {return min(a);}
int max(... int[] a) {return max(a);}

real min(... real[] a) {return min(a);}
real max(... real[] a) {return max(a);}

bool finite(real x)
{
  return abs(x) < infinity;
}

bool finite(pair z)
{
  return abs(z.x) < infinity && abs(z.y) < infinity;
}

bool finite(triple v)
{
  return abs(v.x) < infinity && abs(v.y) < infinity && abs(v.z) < infinity;
}

restricted file stdin=input();
restricted file stdout=output();

void none(file file) {}
void endl(file file) {write(file,'\n',flush);}
void newl(file file) {write(file,'\n');}
void DOSendl(file file) {write(file,'\r\n',flush);}
void DOSnewl(file file) {write(file,'\r\n');}
void tab(file file) {write(file,'\t');}
void comma(file file) {write(file,',');}
typedef void suffix(file);

// Used by interactive write to warn that the outputted type is the resolution
// of an overloaded name.
void overloadedMessage(file file) {
  write(file,'    <overloaded>');
  endl(file);
}

void write(suffix suffix=endl) {suffix(stdout);}
void write(file file, suffix suffix=none) {suffix(file);}

path box(pair a, pair b)
{
  return a--(b.x,a.y)--b--(a.x,b.y)--cycle;
}

restricted path unitsquare=box((0,0),(1,1));

restricted path unitcircle=E..N..W..S..cycle;
restricted real circleprecision=0.0006;

restricted transform invert=reflect((0,0),(1,0));

restricted pen defaultpen;

// A type that takes on one of the values true, false, or default.
struct bool3 {
  bool value;
  bool set;
}

void write(file file, string s="", bool3 b, suffix suffix=none)
{
  if(b.set) write(b.value,suffix);
  else write("default",suffix);
}

void write(string s="", bool3 b, suffix suffix=endl) 
{
  write(stdout,s,b,suffix);
}

restricted bool3 default;

bool operator cast(bool3 b)
{
  return b.set && b.value;
}

bool3 operator cast(bool b)
{
  bool3 B;
  B.value=b;
  B.set=true;
  return B;
}

bool operator == (bool3 a, bool3 b) 
{
  return a.set == b.set && (!a.set || (a.value == b.value));
}

bool operator != (bool3 a, bool3 b) 
{
  return a.set != b.set || (a.set && (a.value != b.value));
}

bool operator == (bool3 a, bool b) 
{
  return a.set && a.value == b;
}

bool operator != (bool3 a, bool b) 
{
  return !a.set || a.value != b;
}

bool operator == (bool a, bool3 b) 
{
  return b.set && b.value == a;
}

bool operator != (bool a, bool3 b) 
{
  return !b.set || b.value != a;
}

bool[] operator cast(bool3[] b)
{
  return sequence(new bool(int i) {return b[i];},b.length);
}

bool3[] operator cast(bool[] b)
{
  return sequence(new bool3(int i) {return b[i];},b.length);
}
