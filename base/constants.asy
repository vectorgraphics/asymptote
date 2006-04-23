// Reduced for tension atleast infinity
real infinity=sqrt(0.25*realMax);
pair Infinity=(infinity,infinity);

real inches=72;
real inch=inches;
real cm=inches/2.540005;
real mm=0.1cm;
real bp=1;	   // A PostScript point.
real pt=72.0/72.27; // A TeX pt; slightly smaller than a PostScript bp.
pair I=(0,1);

pair up=(0,1);
pair down=(0,-1);
pair right=(1,0);
pair left=(-1,0);

pair E=(1,0);
pair N=(0,1);
pair W=(-1,0);
pair S=(0,-1);

pair NE=unit(N+E);
pair NW=unit(N+W);
pair SW=unit(S+W);
pair SE=unit(S+E);

pair ENE=unit(E+NE);
pair NNE=unit(N+NE);
pair NNW=unit(N+NW);
pair WNW=unit(W+NW);
pair WSW=unit(W+SW);
pair SSW=unit(S+SW);
pair SSE=unit(S+SE);
pair ESE=unit(E+SE);
  
// Global parameters:
public real labelmargin=0.3;
public real arrowlength=0.75cm;
public real arrowfactor=15;
public real arrowangle=15;
public real arcarrowfactor=0.5*arrowfactor;
public real arcarrowangle=2*arrowangle;
public real barfactor=arrowfactor;
public real dotfactor=6;

public real legendlinelength=50;
public real legendskip=1.1;
public real legendmargin=10;

public string defaultfilename;
public string defaultformat="$%.4g$";

bool Aspect=true;
bool IgnoreAspect=false;

bool Above=true;
bool Below=false;

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

file stdin=input("");
file stdout;

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
public real circleprecision=0.0006;

transform invert=reflect((0,0),(1,0));
