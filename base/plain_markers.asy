real legendlinelength=50;
real legendhskip=1.2;
real legendvskip=legendhskip;
real legendmargin=10;
real legendmaxrelativewidth=1;

// Return a unit polygon with n sides.
path polygon(int n) 
{
  guide g;
  for(int i=0; i < n; ++i) g=g--expi(2pi*(i+0.5)/n-0.5*pi);
  return g--cycle;
}

// Return a unit n-point cyclic cross, with optional inner radius r and
// end rounding.
path cross(int n, bool round=true, real r=0) 
{
  assert(n > 1);
  real r=min(r,1);
  real theta=pi/n;
  real s=sin(theta);
  real c=cos(theta);
  pair z=(c,s);
  transform mirror=reflect(0,z);
  pair p1=(r,0);
  path elementary;
  if(round) {
    pair e1=p1+z*max(1-r*(s+c),0);
    elementary=p1--e1..(c,s)..mirror*e1--mirror*p1;
  } else {
    pair p2=p1+z*(max(sqrt(1-(r*s)^2)-r*c),0);
    elementary=p1--p2--mirror*p2--mirror*p1;
  }

  guide g;
  real step=360/n;
  for(int i=0; i < n; ++i)
    g=g--rotate(i*step-90)*elementary;

  return g--cycle;
}

path[] plus=(-1,0)--(1,0)^^(0,-1)--(0,1);

typedef void markroutine(picture pic=currentpicture, frame f, path g);

// On picture pic, add frame f about every node of path g.
void marknodes(picture pic=currentpicture, frame f, path g) {
  for(int i=0; i < size(g); ++i)
    add(pic,f,point(g,i));
}

// On picture pic, add n copies of frame f to path g, evenly spaced in
// arclength.
// If rotated=true, the frame will be rotated by the angle of the tangent
// to the path at the points where the frame will be added.
// If centered is true, center the frames within n evenly spaced arclength
// intervals.
markroutine markuniform(bool centered=false, int n, bool rotated=false) {
  return new void(picture pic=currentpicture, frame f, path g) {
    if(n <= 0) return;
    void add(real x) {
      real t=reltime(g,x);
      add(pic,rotated ? rotate(degrees(dir(g,t)))*f : f,point(g,t));
    }
    if(centered) {
      real width=1/n;
      for(int i=0; i < n; ++i) add((i+0.5)*width);
    } else {
      if(n == 1) add(0.5);
      else {
        real width=1/(n-1);
        for(int i=0; i < n; ++i)
          add(i*width);
      }
    }
  };
}

// On picture pic, add frame f at points z(t) for n evenly spaced values of
// t in [a,b].
markroutine markuniform(pair z(real t), real a, real b, int n)
{
  return new void(picture pic=currentpicture, frame f, path) {
    real width=b-a;
    for(int i=0; i <= n; ++i) {
      add(pic,f,z(a+i/n*width));
    }
  };
}

struct marker {
  frame f;
  bool above=true;
  markroutine markroutine=marknodes;
  void mark(picture pic=currentpicture, path g) {
    markroutine(pic,f,g);
  };
}
  
marker marker(frame f=newframe, markroutine markroutine=marknodes,
              bool above=true) 
{
  marker m=new marker;
  m.f=f;
  m.above=above;
  m.markroutine=markroutine;
  return m;
}

marker marker(path[] g, markroutine markroutine=marknodes, pen p=currentpen,
              filltype filltype=NoFill, bool above=true)
{
  frame f;
  filltype.fill(f,g,p);
  return marker(f,markroutine,above);
}

// On picture pic, add path g with opacity thinning about every node.
marker markthin(path g, pen p=currentpen,
                real thin(real fraction)=new real(real x) {return x^2;},
                filltype filltype=NoFill) {
  marker M=new marker;
  M.above=true;
  filltype.fill(M.f,g,p);
  real factor=1/abs(size(M.f));
  M.markroutine=new void(picture pic=currentpicture, frame, path G) {
    transform t=pic.calculateTransform();
    int n=size(G);
    for(int i=0; i < n; ++i) {
      pair z=point(G,i);
      frame f;
      real fraction=1;
      if(i > 0) fraction=min(fraction,abs(t*(z-point(G,i-1)))*factor);
      if(i < n-1) fraction=min(fraction,abs(t*(point(G,i+1)-z))*factor);
      filltype.fill(f,g,p+opacity(thin(fraction)));
      add(pic,f,point(G,i));
    }
  };
  return M;
}

marker nomarker;

real circlescale=0.85;

path[] MarkPath={scale(circlescale)*unitcircle,
                 polygon(3),polygon(4),polygon(5),invert*polygon(3),
                 cross(4),cross(6)};

marker[] Mark=sequence(new marker(int i) {return marker(MarkPath[i]);},
                       MarkPath.length);

marker[] MarkFill=sequence(new marker(int i) {return marker(MarkPath[i],Fill);},
                           MarkPath.length-2);

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
  if(!legenditem.above && !empty(legenditem.mark))
    marknodes(pic,legenditem.mark,interp(z1,z2,0.5));
  if(linelength > 0)
    Draw(pic,z1--z2,legenditem.p);
  if(legenditem.above && !empty(legenditem.mark))
    marknodes(pic,legenditem.mark,interp(z1,z2,0.5));
  if(legenditem.plabel != invisible)
    label(pic,legenditem.label,z2,E,legenditem.plabel);
  else
    label(pic,legenditem.label,z2,E,currentpen);
  return pic;
}

picture legend(Legend[] Legend, int perline=1, real linelength,
               real hskip, real vskip, real maxwidth=0, real maxheight=0,
               bool hstretch=false, bool vstretch=false)
{
  if(maxwidth <= 0) hstretch=false;
  if(maxheight <= 0) vstretch=false;
  if(Legend.length <= 1) vstretch=hstretch=false;

  picture inset;
  size(inset,0,0,IgnoreAspect);

  if(Legend.length == 0)
    return inset;

  // Check for legend entries with lines: 
  bool bLineEntriesAvailable=false;
  for(int i=0; i < Legend.length; ++i) {
    if(Legend[i].p != invisible) {
      bLineEntriesAvailable=true;
      break;
    }
  }

  real markersize=0;
  for(int i=0; i < Legend.length; ++i)
    markersize=max(markersize,size(Legend[i].mark).x);

  // If no legend has a line, set the line length to zero
  if(!bLineEntriesAvailable)
    linelength=0;

  linelength=max(linelength,markersize*(linelength == 0 ? 1 : 2));

  // Get the maximum dimensions per legend entry;
  // calculate line length for a one-line legend
  real heightPerEntry=0;
  real widthPerEntry=0;
  real totalwidth=0;
  for(int i=0; i < Legend.length; ++i) {
    picture pic=legenditem(Legend[i],linelength);
    pair lambda=size(pic);
    heightPerEntry=max(heightPerEntry,lambda.y);
    widthPerEntry=max(widthPerEntry,lambda.x);
    if(Legend[i].p != invisible)
      totalwidth += lambda.x;
    else {
      // Legend entries without leading line need less space in one-line legends
      picture pic=legenditem(Legend[i],0);
      totalwidth += size(pic).x;
    }
  }
  // Does everything fit into one line? 
  if(((perline < 1) || (perline >= Legend.length)) && 
     (maxwidth >= totalwidth+(totalwidth/Legend.length)*
      (Legend.length-1)*(hskip-1))) {
    // One-line legend
    real currPosX=0;
    real itemDistance;
    if(hstretch)
      itemDistance=(maxwidth-totalwidth)/(Legend.length-1);
    else
      itemDistance=(totalwidth/Legend.length)*(hskip-1);
    for(int i=0; i < Legend.length; ++i) {
      picture pic=legenditem(Legend[i],
                             Legend[i].p == invisible ? 0 : linelength);
      add(inset,pic,(currPosX,0));
      currPosX += size(pic).x+itemDistance;
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

    if(perline <= 1) hstretch=false;
    if(hstretch) hskip=(maxwidth/widthPerEntry-perline)/(perline-1)+1;
    if(vstretch) {
      int rows=ceil(Legend.length/perline);
      vskip=(maxheight/heightPerEntry-rows)/(rows-1)+1;
    }

    if(hstretch && (perline == 1)) {
      Draw(inset,(0,0)--(maxwidth,0),invisible());
      for(int i=0; i < Legend.length; ++i)
        add(inset,legenditem(Legend[i],linelength),
            (0.5*(maxwidth-widthPerEntry),
             -quotient(i,perline)*heightPerEntry*vskip));
    } else
      for(int i=0; i < Legend.length; ++i)
        add(inset,legenditem(Legend[i],linelength),
            ((i%perline)*widthPerEntry*hskip,
             -quotient(i,perline)*heightPerEntry*vskip));
  }

  return inset;
}

frame legend(picture pic=currentpicture, int perline=1,
             real xmargin=legendmargin, real ymargin=xmargin,
             real linelength=legendlinelength,
             real hskip=legendhskip, real vskip=legendvskip,
             real maxwidth=perline == 0 ? 
             legendmaxrelativewidth*size(pic).x : 0, real maxheight=0,
	     bool hstretch=false, bool vstretch=false, pen p=currentpen)
{
  frame F;
  if(pic.legend.length == 0) return F;
  F=legend(pic.legend,perline,linelength,hskip,vskip,
           max(maxwidth-2xmargin,0),
           max(maxheight-2ymargin,0),
           hstretch,vstretch).fit();
  box(F,xmargin,ymargin,p);
  return F;
}

pair[] pairs(real[] x, real[] y)
{
  if(x.length != y.length) abort("arrays have different lengths");
  return sequence(new pair(int i) {return (x[i],y[i]);},x.length);
}

filltype dotfilltype = Fill;

void dot(frame f, pair z, pen p=currentpen, filltype filltype=dotfilltype)
{
  if(filltype == Fill)
    draw(f,z,dotsize(p)+p);
  else {
    real s=0.5*(dotsize(p)-linewidth(p));  
    if(s <= 0) return;
    path g=shift(z)*scale(s)*unitcircle;
    begingroup(f);
    filltype.fill(f,g,p);
    draw(f,g,p);
    endgroup(f);
  }
}

void dot(picture pic=currentpicture, pair z, pen p=currentpen,
         filltype filltype=dotfilltype)
{
  pic.add(new void(frame f, transform t) {
      dot(f,t*z,p,filltype);
    },true);
  pic.addPoint(z,dotsize(p)+p);
}

void dot(picture pic=currentpicture, Label L, pair z, align align=NoAlign,
         string format=defaultformat, pen p=currentpen, filltype filltype=dotfilltype)
{
  Label L=L.copy();
  L.position(z);
  if(L.s == "") {
    if(format == "") format=defaultformat;
    L.s="("+format(format,z.x)+","+format(format,z.y)+")";
  }
  L.align(align,E);
  L.p(p);
  dot(pic,z,p,filltype);
  add(pic,L);
}

void dot(picture pic=currentpicture, Label[] L=new Label[], pair[] z,
	 align align=NoAlign, string format=defaultformat, pen p=currentpen,
	 filltype filltype=dotfilltype)
{
  int stop=min(L.length,z.length);
  for(int i=0; i < stop; ++i)
    dot(pic,L[i],z[i],align,format,p,filltype);
  for(int i=stop; i < z.length; ++i)
    dot(pic,z[i],p,filltype);
}

void dot(picture pic=currentpicture, Label[] L=new Label[],
	 explicit path g, align align=RightSide, string format=defaultformat,
	 pen p=currentpen, filltype filltype=dotfilltype)
{
  int n=size(g);
  int stop=min(L.length,n);
  for(int i=0; i < stop; ++i)
    dot(pic,L[i],point(g,i),-sgn(align.dir.x)*I*dir(g,i),format,p,filltype);
  for(int i=stop; i < n; ++i)
    dot(pic,point(g,i),p,filltype);
}

void dot(picture pic=currentpicture, path[] g, pen p=currentpen,
         filltype filltype=dotfilltype)
{
  for(int i=0; i < g.length; ++i)
    dot(pic,g[i],p,filltype);
}

void dot(picture pic=currentpicture, Label L, pen p=currentpen,
         filltype filltype=dotfilltype)
{
  dot(pic,L,L.position,p,filltype);
}

// A dot in a frame.
frame dotframe(pen p=currentpen, filltype filltype=dotfilltype)
{
  frame f;
  dot(f,(0,0),p,filltype);
  return f;
}

frame dotframe=dotframe();

marker dot(pen p=currentpen, filltype filltype=dotfilltype)
{
  return marker(dotframe(p,filltype));
}

marker dot=dot();
    
