real legendlinelength=50;
real legendhskip=1.2;
real legendvskip=legendhskip;
real legendmargin=10;
real legendmaxrelativewidth=1;

// Return a unit polygon with n sides
guide polygon(int n) 
{
  guide g;
  for(int i=0; i < n; ++i) g=g--expi(2pi*(i+0.5)/n-0.5*pi);
  return g--cycle;
}

// Return an n-point unit cross
path[] cross(int n) 
{
  path[] g;
  for(int i=0; i < n; ++i) g=g^^(0,0)--expi(2pi*(i+0.5)/n-0.5*pi);
  return g;
}

path[] plus=(-1,0)--(1,0)^^(0,-1)--(0,1);

typedef void markroutine(picture pic=currentpicture, path g, frame f);

// On picture pic, add to every node of path g the frame f.
void marknodes(picture pic=currentpicture, path g, frame f) {
  for(int i=0; i <= length(g); ++i)
    add(pic,f,point(g,i));
}

// On picture pic, add to path g the frame f, evenly spaced in arclength.
// If rotated=true, the frame will be rotated by the angle of the tangent
// to the path at the points where the frame will be added.
markroutine markuniform(int n, bool rotated=false) {
  rotated=true;
  return new void(picture pic=currentpicture, path g, frame f) {
    if(n == 0) return;
    void add(real x) {
      real t=reltime(g,x);
      add(pic,rotated ? rotate(degrees(dir(g,t)))*f : f,point(g,t));
    }
    if(n == 1) add(0.5);
    else {
      real width=1/(n-1);
      for(int i=0; i < n; ++i)
	add(i*width);
    }
  };
}

struct marker {
  frame f;
  bool put=Above;
  markroutine markroutine=marknodes;
  void mark(picture pic, path g) {
    markroutine(pic,g,f);
  };
}
  
marker operator init() {return new marker;}
  
marker marker(frame f, markroutine markroutine=marknodes, bool put=Above) 
{
  marker m=new marker;
  m.f=f;
  m.put=put;
  m.markroutine=markroutine;
  return m;
}

marker marker(path[] g, markroutine markroutine=marknodes, pen p=currentpen,
              filltype filltype=NoFill, bool put=Above)
{
  frame f;
  filltype(f,g,p);
  return marker(f,markroutine,put);
}

marker nomarker;

real circlescale=0.85;

marker[] Mark={
  marker(scale(circlescale)*unitcircle),
  marker(polygon(3)),marker(polygon(4)),
  marker(polygon(5)),marker(invert*polygon(3)),
  marker(cross(4)),marker(cross(6))
};

marker[] MarkFill={
  marker(scale(circlescale)*unitcircle,Fill),marker(polygon(3),Fill),
  marker(polygon(4),Fill),marker(polygon(5),Fill),
  marker(invert*polygon(3),Fill)
};

marker Mark(int n) 
{
  n=n % (Mark.length+MarkFill.length);
  if(n < Mark.length) return Mark[n];
  else return MarkFill[n-Mark.length];
}

picture legenditem(Legend legenditem, real linelength)
{
  picture pic;
  pair z1=(0,0);
  pair z2=z1+(linelength,0);
  if(!legenditem.put && !empty(legenditem.mark))
    marknodes(pic,interp(z1,z2,0.5),legenditem.mark);
  if(linelength > 0)
    Draw(pic,z1--z2,legenditem.p);
  if(legenditem.put && !empty(legenditem.mark))
    marknodes(pic,interp(z1,z2,0.5),legenditem.mark);
  if(legenditem.plabel != invisible)
    label(pic,legenditem.label,z2,E,legenditem.plabel);
  else
    label(pic,legenditem.label,z2,E,currentpen);
  return pic;
}

picture legend(Legend[] Legend, int perline=1, real linelength,
               real hskip, real vskip, real maxwidth=0)
{
  picture inset;
  size(inset,0,0,IgnoreAspect);

  if(Legend.length == 0)
    return inset;

  // Check for legend entries with lines: 
  bool bLineEntriesAvailable=false;
  for(int i=0; i < Legend.length; ++i)
    if(Legend[i].p != invisible)
      bLineEntriesAvailable=true;
  // If no legend has a line, set the line length to zero
  if(!bLineEntriesAvailable)
    linelength=0;

  // Get the maximum dimensions per legend entry;
  // calculate line length for a one-line legend
  real heightPerEntry=0;
  real widthPerEntry=0;
  real totalwidth=0;
  for(int i=0; i < Legend.length; ++i) {
    picture pic=legenditem(Legend[i],linelength);
    heightPerEntry=max(heightPerEntry,max(pic).y-min(pic).y);
    widthPerEntry=max(widthPerEntry,max(pic).x-min(pic).x);
    if(Legend[i].p != invisible)
      totalwidth += max(pic).x-min(pic).x;
    else {
      // Legend entries without leading line need less space in one-line legends
      picture pic=legenditem(Legend[i],0);
      totalwidth += max(pic).x-min(pic).x;
    }
  }
  // Does everything fit into one line? 
  if(((perline < 1) || (perline >= Legend.length)) && 
     (maxwidth >= totalwidth+(totalwidth/Legend.length)*
      (Legend.length-1)*(hskip-1))) {
    // One-line legend
    real currPosX=0;
    real itemDistance=(totalwidth/Legend.length)*(hskip-1);
    for(int i=0; i < Legend.length; ++i) {
      picture pic=legenditem(Legend[i],
                             Legend[i].p == invisible ? 0 : linelength);
      add(inset,pic,(currPosX,0));
      currPosX += max(pic).x-min(pic).x+itemDistance;
    }
  } else {
    // multiline legend
    if(maxwidth > 0) {
      int maxperline=floor(maxwidth/(widthPerEntry*hskip));
      if((perline < 1) || (perline > maxperline))
        perline=maxperline;
    }
    if(perline < 1) // This means: maxwidth < widthPerEntry
      perline=1;
 
    for(int i=0; i < Legend.length; ++i)
      add(inset,legenditem(Legend[i],linelength),
          ((i%perline)*widthPerEntry*hskip,
           -floor(i/perline)*heightPerEntry*vskip));
  }

  return inset;
}

frame legend(picture pic=currentpicture, int perline=1,
             real xmargin=legendmargin, real ymargin=xmargin,
             real linelength=legendlinelength,
             real hskip=legendhskip, real vskip=legendvskip,
             real maxwidth=perline == 0 ?
             legendmaxrelativewidth*(max(pic).x-min(pic).x) : 0,
             pen p=currentpen)
{
  frame F;
  if(pic.legend.length == 0) return F;
  F=legend(pic.legend,perline,linelength,hskip,vskip,maxwidth).fit();
  box(F,xmargin,ymargin,p);
  return F;
}

void dot(frame f, pair z, pen p=currentpen)
{
  draw(f,z,dotsize(p)+p);
}

void dot(picture pic=currentpicture, pair z, pen p=currentpen)
{
  Draw(pic,z,dotsize(p)+p);
}

void dot(picture pic=currentpicture, pair[] z, pen p=currentpen)
{
  for(int i=0; i < z.length; ++i) dot(pic,z[i],p);
}

void dot(picture pic=currentpicture, explicit path g, pen p=currentpen)
{
  for(int i=0; i <= length(g); ++i) dot(pic,point(g,i),p);
}

void dot(picture pic=currentpicture, path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) dot(pic,g[i],p);
}

void dot(picture pic=currentpicture, Label L, pair z, align align=NoAlign,
         string format=defaultformat, pen p=currentpen)
{
  Label L=L.copy();
  L.position(z);
  if(L.s == "") {
    if(format == "") format=defaultformat;
    L.s="("+format(format,z.x)+","+format(format,z.y)+")";
  }
  L.align(align,E);
  L.p(p);
  dot(pic,z,p);
  add(pic,L);
}

void dot(picture pic=currentpicture, Label L, pen p=currentpen)
{
  dot(pic,L,L.position,p);
}
