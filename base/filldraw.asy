// Draw path g on frame f with user-constructed pen p.
void makedraw(frame f, path g, pen p)
{
  path n=nib(p);
  if(size(g) == 1) fill(f,shift(point(g,0))*n,p);
  static real epsilon=1000*realEpsilon;
  real stop=length(g)-epsilon;
  int N=length(n);
  pair n0=point(n,0);
  for(int i=0; i < N; ++i) {
    pair n1=point(n,i+1);
    pair dir=n1-n0;
    real tm=dirtime(g,-dir);
    real tp=dirtime(g,dir);
    if(tm > epsilon && tm < stop) {
      makedraw(f,subpath(g,0,tm),p);
      makedraw(f,subpath(g,tm,length(g)),p);
      return;
    }	
    if(tp > epsilon && tp < stop) {
      makedraw(f,subpath(g,0,tp),p);
      makedraw(f,subpath(g,tp,length(g)),p);
      return;
    }
    n0=n1;
  }
  pair n0=point(n,0);
  for(int i=0; i < N; ++i) {
    pair n1=point(n,i+1);
    fill(f,shift(n0)*g--shift(n1)*reverse(g)--cycle,p);
    n0=n1;
  }
}

void draw(frame f, path g, pen p=currentpen)
{
  if(size(nib(p)) == 0) _draw(f,g,p);
  else {
    begingroup(f);
    makedraw(f,g,p);
    endgroup(f);
  }
}

void draw(frame f, explicit path[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) draw(f,g[i],p);
}

void fill(frame f, path[] g)
{
  fill(f,g,currentpen);
}

void filldraw(frame f, path[] g, pen fillpen=currentpen,
	      pen drawpen=currentpen)
{
  begingroup(f);
  fill(f,g,fillpen);
  draw(f,g,drawpen);
  endgroup(f);
}

void unfill(frame f, path[] g)
{
  static pair margin=(0.5,0.5);
  clip(f,box(min(f)-margin,max(f)+margin)^^g,evenoddoverlap);
}

typedef void filltype(frame, path, pen);
void filltype(frame, path, pen) {}

path margin(path g, real xmargin, real ymargin) 
{
  if(xmargin != 0 || ymargin != 0) {
    pair M=max(g);
    pair m=min(g);
    real width=M.x-m.x;
    real height=M.y-m.y;
    real xfactor=(width+2xmargin)/width;
    real yfactor=(height+2ymargin)/height;
    g=xscale(xfactor)*yscale(yfactor)*g;
    g=shift(0.5*(M+m)-0.5*(max(g)+min(g)))*g;
  }   
  return g;
}

filltype Fill(real xmargin=0, real ymargin=xmargin, pen p)
{
  return new void(frame f, path g, pen drawpen) {
    filldraw(f,margin(g,xmargin,ymargin),p == nullpen ? drawpen : p,drawpen);
  };
}

public filltype NoFill=new void(frame f, path g, pen p) {
  draw(f,g,p);
};

filltype Fill(real xmargin=0, real ymargin=0)
{
  return Fill(xmargin,ymargin,nullpen);
}

public filltype Fill=Fill(nullpen);

filltype UnFill(real xmargin=0, real ymargin=xmargin)
{
  return new void(frame f, path g, pen p) {
    unfill(f,margin(g,xmargin,ymargin));
  };
}

public filltype UnFill=UnFill();

// Fill varying radially from penc at the center of the bounding box to
// penr at the edge.
filltype RadialShade(pen penc, pen penr)
{
  return new void(frame f, path g, pen) {
    pair c=(min(g)+max(g))/2;
    radialshade(f,g,penc,c,0,penr,c,abs(max(g)-min(g))/2);
  };
}

// Fill the region in frame dest underneath frame src and return the
// boundary of src.
guide fill(frame dest, frame src, filltype filltype=NoFill, 
	   real xmargin=0, real ymargin=xmargin)
{
  pair z=(xmargin,ymargin);
  guide g=box(min(src)-0.5*min(invisible)-z,max(src)-0.5*max(invisible)+z);
  filltype(dest,g,invisible);
  return g;
}

// Add frame dest to frame src with optional grouping and background fill.
void add(frame dest, frame src, bool group, filltype filltype=NoFill,
	 bool put=Above)
{
  if(put) {
    if(filltype != NoFill) fill(dest,src,filltype);
    if(group) begingroup(dest);
    add(dest,src);
    if(group) endgroup(dest);
  } else {
    if(group) {
      frame f;
      endgroup(f);
      prepend(dest,f);
    }
    prepend(dest,src);
    if(group) {
      frame f;
      begingroup(f);
      prepend(dest,f);
    }
    if(filltype != NoFill) {
      frame f;
      fill(f,src,filltype);
      prepend(dest,f);
    }
  }
}

void add(frame dest, frame src, filltype filltype, bool put=Above)
{
  if(filltype != NoFill) fill(dest,src,filltype);
  (put ? add : prepend)(dest,src);
}

