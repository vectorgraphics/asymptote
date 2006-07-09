restricted real infinity=cbrt(realMax); // Reduced for tension atleast infinity
restricted pair Infinity=(infinity,infinity);

restricted real inches=72;
restricted real inch=inches;
restricted real cm=inches/2.540005;
restricted real mm=0.1cm;
restricted real bp=1;	   // A PostScript point.
restricted real pt=72.0/72.27; // A TeX pt; smaller than a PostScript bp.
restricted pair I=(0,1);

restricted pair up=(0,1);
restricted pair down=(0,-1);
restricted pair right=(1,0);
restricted pair left=(-1,0);

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
  
string defaultfilename;
string defaultformat="$%.4g$";

restricted bool Above=true;
restricted bool Below=false;

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

restricted file stdin=input("");
restricted file stdout;

void none(file file) {}
void endl(file file) {write(file,'\n'); flush(file);}
void tab(file file) {write(file,'\t');}
typedef void suffix(file);

void write(file file=stdout, suffix suffix=endl) {suffix(file);}

guide box(pair a, pair b)
{
  return a--(b.x,a.y)--b--(a.x,b.y)--cycle;
}

guide unitsquare=box((0,0),(1,1));

guide unitcircle=E..N..W..S..cycle;
restricted real circleprecision=0.0006;

restricted transform invert=reflect((0,0),(1,0));
