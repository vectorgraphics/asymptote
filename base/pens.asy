public pen currentpen;

pen nullpen=linewidth(0);

pen solid=linetype("");
pen dotted=linetype("0 4");
pen dashed=linetype("8 8");
pen longdashed=linetype("24 8");
pen dashdotted=linetype("8 8 0 8");
pen longdashdotted=linetype("24 8 0 8");

void defaultpen(real w) {defaultpen(linewidth(w));}
pen operator +(pen p, real w) {return p+linewidth(w);}
pen operator +(real w, pen p) {return linewidth(w)+p;}

pen Dotted=dotted+1.0;
pen Dotted(pen p) {return dotted+2*linewidth(p);}

pen squarecap=linecap(0);
pen roundcap=linecap(1);
pen extendcap=linecap(2);

pen miterjoin=linejoin(0);
pen roundjoin=linejoin(1);
pen beveljoin=linejoin(2);

pen zerowinding=fillrule(0);
pen evenodd=fillrule(1);
pen zerowindingoverlap=fillrule(2);
pen evenoddoverlap=fillrule(3);

pen nobasealign=basealign(0);
pen basealign=basealign(1);

pen invisible=invisible();
pen black=gray(0);
pen white=gray(1);
pen gray=gray(0.5);

pen red=rgb(1,0,0);
pen green=rgb(0,1,0);
pen blue=rgb(0,0,1);

pen cmyk=cmyk(0,0,0,0);
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
pen olive=rgb(0.5,0.5,0);
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
pen darkbrown=darkred;
pen pink=palemagenta;
pen palegrey=palegray;
pen lightgrey=lightgray;
pen mediumgrey=mediumgray;
pen grey=gray;
pen heavygrey=gray;
pen deepgrey=deepgray;
pen darkgrey=darkgray;

pen cmyk(pen p) 
{
  return p+cmyk;
}

real linewidth() 
{
  return linewidth(currentpen);
}

real lineskip() 
{
  return lineskip(currentpen);
}

// Options for handling label overwriting
int Allow=0;
int Suppress=1;
int SuppressQuiet=2;
int Move=3;
int MoveQuiet=4;

public pen[] colorPen={red,blue,green,magenta,cyan,orange,purple,brown,
		       deepblue,deepgreen,chartreuse,fuchsia,lightred,
		       lightblue,black,pink,yellow,gray};

colorPen.cyclic(true);

public pen[] monoPen={solid,dashed,dotted,longdashed,dashdotted,
		      longdashdotted};
monoPen.cyclic(true);

public bool mono=false;

pen Pen(int n) 
{
  return mono ? monoPen[n] : colorPen[n];
}

real dotsize(pen p=currentpen) 
{
  return dotfactor*linewidth(p);
}

real arrowsize(pen p=currentpen) 
{
  return arrowfactor*linewidth(p);
}

real arcarrowsize(pen p=currentpen) 
{
  return arcarrowfactor*linewidth(p);
}

real barsize(pen p=currentpen) 
{
  return barfactor*linewidth(p);
}

pen fontsize(real size) 
{
  return fontsize(size,1.2*size);
}

real fontsize() 
{
  return fontsize(currentpen);
}

real labelmargin(pen p=currentpen)
{
  return labelmargin*fontsize(p);
}

pen interp(pen a, pen b, real t) 
{
  return (1-t)*a+t*b;
}

pen font(string name) 
{
  return fontcommand("\font\ASYfont="+name+"\ASYfont");
}

pen font(string name, real size) 
{
  // Extract size of requested TeX font
  string basesize;
  for(int i=0; i < length(name); ++i) {
    string c=substr(name,i,1);
    if(c >= "0" && c <= "9") basesize += c;
    else if(basesize != "") break;
  }
  return basesize == "" ? font(name) :
    font(name+" scaled "+(string) (1000*size/(int) basesize)); 
}

pen font(string encoding, string family, string series="m", string shape="n") 
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
