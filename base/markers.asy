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
markroutine markuniform(int n) {
  return new void(picture pic=currentpicture, path g, frame f) {
    if(n == 0) return;
    if(n == 1) add(pic,f,relpoint(g,0.5));
    else {
      real width=1/(n-1);
      for(int i=0; i < n; ++i)
	add(pic,f,relpoint(g,i*width));
    }
  };
}

struct marker {
  public frame f;
  public bool put=Above;
  public markroutine markroutine=marknodes;
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
  if(filltype == Fill) fill(f,g,p);
  else draw(f,g,p);
  return marker(f,markroutine,put);
}

marker nomarker;

public real circlescale=0.85;

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

picture legend(Legend[] legend, real length, real skip)
{
  picture inset;
  size(inset,0,0,IgnoreAspect);
  if(legend.length > 0) {
    frame f;
    real height=0;
    for(int i=0; i < legend.length; ++i) {
      Legend L=legend[i];
      frame f;
      draw(f,(0,0),L.p);
      label(f,L.label,(0,0),L.plabel);
      if(!empty(L.mark)) add(f,L.mark,(0,0));
      height=max(height,max(f).y-min(f).y);
    }
    for(int i=0; i < legend.length; ++i) {
      Legend L=legend[i];
      pair z1=(0,-i*height*skip);
      pair z2=z1+length;
      if(!L.put && !empty(L.mark)) marknodes(inset,interp(z1,z2,0.5),L.mark);
      Draw(inset,z1--z2,L.p);
      label(inset,L.label,z2,E,L.plabel);
      if(L.put && !empty(L.mark)) marknodes(inset,interp(z1,z2,0.5),L.mark);
    }
  }
  return inset;
}
  
frame legend(picture pic=currentpicture,
	     real xmargin=legendmargin, real ymargin=xmargin,
	     real length=legendlinelength, real skip=legendskip,
	     pen p=currentpen)
{
  frame F;
  if(pic.legend.length == 0) return F;
  F=legend(pic.legend,length,skip).fit();
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
  if(L.s == "") L.s="("+format(format,z.x)+","+format(format,z.y)+")";
  L.align(align,E);
  L.p(p);
  dot(pic,z,p);
  add(pic,L);
}

void dot(picture pic=currentpicture, Label L, pen p=currentpen)
{
  dot(pic,L,L.position,p);
}
