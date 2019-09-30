real labelmargin=0.3;
real dotfactor=6;

pen solid=linetype(new real[]);
pen dotted=linetype(new real[] {0,4});
pen dashed=linetype(new real[] {8,8});
pen longdashed=linetype(new real[] {24,8});
pen dashdotted=linetype(new real[] {8,8,0,8});
pen longdashdotted=linetype(new real[] {24,8,0,8});

pen linetype(string pattern, real offset=0, bool scale=true, bool adjust=true) 
{
  return linetype((real[]) split(pattern),offset,scale,adjust);
}

void defaultpen(real w) {defaultpen(linewidth(w));}
pen operator +(pen p, real w) {return p+linewidth(w);}
pen operator +(real w, pen p) {return linewidth(w)+p;}

pen Dotted(pen p=currentpen) {return linetype(new real[] {0,3})+2*linewidth(p);}
pen Dotted=Dotted();

restricted pen squarecap=linecap(0);
restricted pen roundcap=linecap(1);
restricted pen extendcap=linecap(2);

restricted pen miterjoin=linejoin(0);
restricted pen roundjoin=linejoin(1);
restricted pen beveljoin=linejoin(2);

restricted pen zerowinding=fillrule(0);
restricted pen evenodd=fillrule(1);

bool interior(int windingnumber, pen fillrule)
{
  return windingnumber != undefined &&
    (fillrule(fillrule) == 1 ? windingnumber % 2 == 1 : windingnumber != 0);
}

restricted pen nobasealign=basealign(0);
restricted pen basealign=basealign(1);

pen invisible=invisible();
pen thin() {return settings.thin ? linewidth(0) : defaultpen;}
pen thick(pen p=currentpen) {return linewidth(linewidth(p));}
pen nullpen=linewidth(0)+invisible;

pen black=gray(0);
pen white=gray(1);
pen gray=gray(0.5);

pen red=rgb(1,0,0);
pen green=rgb(0,1,0);
pen blue=rgb(0,0,1);

pen Cyan=cmyk(1,0,0,0);
pen Magenta=cmyk(0,1,0,0);
pen Yellow=cmyk(0,0,1,0);
pen Black=cmyk(0,0,0,1);

pen cyan=rgb(0,1,1);
pen magenta=rgb(1,0,1);
pen yellow=rgb(1,1,0);

pen palered=rgb(1,0.75,0.75);
pen palegreen=rgb(0.75,1,0.75);
pen paleblue=rgb(0.75,0.75,1);
pen palecyan=rgb(0.75,1,1);
pen palemagenta=rgb(1,0.75,1);
pen paleyellow=rgb(1,1,0.75);
pen palegray=gray(0.95);

pen lightred=rgb(1,0.5,0.5);
pen lightgreen=rgb(0.5,1,0.5);
pen lightblue=rgb(0.5,0.5,1);
pen lightcyan=rgb(0.5,1,1);
pen lightmagenta=rgb(1,0.5,1);
pen lightyellow=rgb(1,1,0.5);
pen lightgray=gray(0.9);

pen mediumred=rgb(1,0.25,0.25);
pen mediumgreen=rgb(0.25,1,0.25);
pen mediumblue=rgb(0.25,0.25,1);
pen mediumcyan=rgb(0.25,1,1);
pen mediummagenta=rgb(1,0.25,1);
pen mediumyellow=rgb(1,1,0.25);
pen mediumgray=gray(0.75);

pen heavyred=rgb(0.75,0,0);
pen heavygreen=rgb(0,0.75,0);
pen heavyblue=rgb(0,0,0.75);
pen heavycyan=rgb(0,0.75,0.75);
pen heavymagenta=rgb(0.75,0,0.75);
pen lightolive=rgb(0.75,0.75,0);
pen heavygray=gray(0.25);

pen deepred=rgb(0.5,0,0);
pen deepgreen=rgb(0,0.5,0);
pen deepblue=rgb(0,0,0.5);
pen deepcyan=rgb(0,0.5,0.5);
pen deepmagenta=rgb(0.5,0,0.5);
pen deepyellow=rgb(0.5,0.5,0);
pen deepgray=gray(0.1);

pen darkred=rgb(0.25,0,0);
pen darkgreen=rgb(0,0.25,0);
pen darkblue=rgb(0,0,0.25);
pen darkcyan=rgb(0,0.25,0.25);
pen darkmagenta=rgb(0.25,0,0.25);
pen darkolive=rgb(0.25,0.25,0);
pen darkgray=gray(0.05);

pen orange=rgb(1,0.5,0);
pen fuchsia=rgb(1,0,0.5);

pen chartreuse=rgb(0.5,1,0);
pen springgreen=rgb(0,1,0.5);

pen purple=rgb(0.5,0,1);
pen royalblue=rgb(0,0.5,1);

// Synonyms:

pen salmon=lightred;
pen brown=deepred;
pen olive=deepyellow;
pen darkbrown=darkred;
pen pink=palemagenta;
pen palegrey=palegray;
pen lightgrey=lightgray;
pen mediumgrey=mediumgray;
pen grey=gray;
pen heavygrey=heavygray;
pen deepgrey=deepgray;
pen darkgrey=darkgray;

// Options for handling label overwriting
restricted int Allow=0;
restricted int Suppress=1;
restricted int SuppressQuiet=2;
restricted int Move=3;
restricted int MoveQuiet=4;

pen[] colorPen={red,blue,green,magenta,cyan,orange,purple,brown,
                deepblue,deepgreen,chartreuse,fuchsia,lightred,
                lightblue,black,pink,yellow,gray};

colorPen.cyclic=true;

pen[] monoPen={solid,dashed,dotted,longdashed,dashdotted,
               longdashdotted};
monoPen.cyclic=true;

pen Pen(int n) 
{
  return (settings.gray || settings.bw) ? monoPen[n] : colorPen[n];
}

pen Pentype(int n)
{
  return (settings.gray || settings.bw) ? monoPen[n] : monoPen[n]+colorPen[n];
}

real dotsize(pen p=currentpen) 
{
  return dotfactor*linewidth(p);
}

pen fontsize(real size) 
{
  return fontsize(size,1.2*size);
}

real labelmargin(pen p=currentpen)
{
  return labelmargin*fontsize(p);
}

void write(file file=stdout, string s="", pen[] p)
{
  for(int i=0; i < p.length; ++i)
    write(file,s,p[i],endl);
}

void usetypescript(string s, string encoding="")
{
  string s="\usetypescript["+s+"]";
  if(encoding != "") s +="["+encoding+"]";
  texpreamble(s);
}

pen font(string name, string options="") 
{
  // Work around misalignment in ConTeXt switchtobodyfont if font is not found.
  return fontcommand(settings.tex == "context" ?
                     "\switchtobodyfont["+name+
                     (options == "" ? "" : ","+options)+
                     "]\removeunwantedspaces" :
                     "\font\ASYfont="+name+"\ASYfont");
}

pen font(string name, real size, string options="") 
{
  string s=(string) (size/pt)+"pt";
  if(settings.tex == "context")
    return fontsize(size)+font(name+","+s,options);
  return fontsize(size)+font(name+" at "+s);
}

pen font(string encoding, string family, string series, string shape) 
{
  return fontcommand("\usefont{"+encoding+"}{"+family+"}{"+series+"}{"+shape+
                     "}");
}

pen AvantGarde(string series="m", string shape="n")
{
  return font("OT1","pag",series,shape);
}
pen Bookman(string series="m", string shape="n")
{
  return font("OT1","pbk",series,shape);
}
pen Courier(string series="m", string shape="n")
{
  return font("OT1","pcr",series,shape);
}
pen Helvetica(string series="m", string shape="n")
{
  return font("OT1","phv",series,shape);
}
pen NewCenturySchoolBook(string series="m", string shape="n")
{
  return font("OT1","pnc",series,shape);
}
pen Palatino(string series="m", string shape="n")
{
  return font("OT1","ppl",series,shape);
}
pen TimesRoman(string series="m", string shape="n")
{
  return font("OT1","ptm",series,shape);
}
pen ZapfChancery(string series="m", string shape="n")
{
  return font("OT1","pzc",series,shape);
}
pen Symbol(string series="m", string shape="n")
{
  return font("OT1","psy",series,shape);
}
pen ZapfDingbats(string series="m", string shape="n")
{
  return font("OT1","pzd",series,shape);
}

pen squarepen=makepen(shift(-0.5,-0.5)*unitsquare);

struct hsv {
  real h;
  real v;
  real s;
  void operator init(real h, real s, real v) {
    this.h=h;
    this.s=s;
    this.v=v;
  }
  void operator init(pen p) {
    real[] c=colors(rgb(p));
    real r=c[0];
    real g=c[1];
    real b=c[2];
    real M=max(r,g,b);
    real m=min(r,g,b);
    if(M == m) this.h=0;
    else {
      real denom=1/(M-m);
      if(M == r) {
	this.h=60*(g-b)*denom;
	if(g < b) h += 360;
      } else if(M == g) {
	this.h=60*(b-r)*denom+120;
      } else
	this.h=60*(r-g)*denom+240;
    }
    this.s=M == 0 ? 0 : 1-m/M;
    this.v=M;
  }
  // return an rgb pen corresponding to h in [0,360) and s and v in [0,1].
  pen rgb() {
    real H=(h % 360)/60;
    int i=floor(H) % 6;
    real f=H-i;
    real[] V={v,v*(1-s),v*(1-(i % 2 == 0 ? 1-f : f)*s)};
    int[] a={0,2,1,1,2,0};
    int[] b={2,0,0,2,1,1};
    int[] c={1,1,2,0,0,2};
    return rgb(V[a[i]],V[b[i]],V[c[i]]);
  }
}

pen operator cast(hsv hsv)
{
  return hsv.rgb();
}

hsv operator cast(pen p)
{
  return hsv(p);
}

real[] rgba(pen p)
{
  real[] a=colors(rgb(p));
  a.push(opacity(p));
  return a;
}

pen rgba(real[] a)
{
  return rgb(a[0],a[1],a[2])+opacity(a[3]);
}

// Return a pen corresponding to a given 6-character RGB hexadecimal string.
pen rgb(string s) 
{
  int offset=substr(s,0,1) == '#' ? 1 : 0;
  real value(string s, int i) {return hex(substr(s,2i+offset,2))/255;}
  return rgb(value(s,0),value(s,1),value(s,2));
}

pen RGB(int r, int g, int b)
{
  return rgb(r/255,g/255,b/255);
}

pen[] operator +(pen[] a, pen b)
{
  return sequence(new pen(int i) {return a[i]+b;},a.length);
}

pen[] operator +(pen a, pen[] b)
{
  return sequence(new pen(int i) {return a+b[i];},b.length);
}

// Interpolate an array of pens in rgb space using by default their minimum
// opacity.
pen mean(pen[] p, real opacity(real[])=min)
{
  if(p.length == 0) return nullpen;
  real[] a=rgba(p[0]);
  real[] t=new real[p.length];
  t[0]=a[3];
  for(int i=1; i < p.length; ++i) {
    real[] b=rgba(p[i]);
    a += b;
    t[i]=b[3];
  }
  a /= p.length;
  return rgb(a[0],a[1],a[2])+opacity(opacity(t));
}

pen[] mean(pen[][] palette, real opacity(real[])=min)
{
  return sequence(new pen(int i) {return mean(palette[i],opacity);},
		  palette.length);
}
