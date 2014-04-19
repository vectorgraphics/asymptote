// Draw path g on frame f with user-constructed pen p.
void makedraw(frame f, path g, pen p, int depth=mantissaBits)
{
  if(depth == 0) return;
  --depth;
  
  path n=nib(p);
  for(int i=0; i < size(g); ++i)
    fill(f,shift(point(g,i))*n,p);

  static real epsilon=1000*realEpsilon;
  int L=length(g);
  real stop=L-epsilon;
  int N=length(n);
  pair first=point(n,0);
  pair n0=first;

  for(int i=0; i < N; ++i) {
    pair n1=point(n,i+1);
    pair dir=unit(n1-n0);
    real t=dirtime(g,-dir)-epsilon;
    if(straight(g,(int) t)) t=ceil(t);
    if(t > epsilon && t < stop) {
      makedraw(f,subpath(g,0,t),p,depth);
      makedraw(f,subpath(g,t,L),p,depth);
      return;
    }
    real t=dirtime(g,dir);
    if(straight(g,(int) t)) t=ceil(t);
    if(t > epsilon && t < stop) {
      makedraw(f,subpath(g,0,t),p,depth);
      makedraw(f,subpath(g,t,L),p,depth);
      return;
    }
    n0=n1;
  }

  n0=first;
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

void draw(frame f, guide[] g, pen p=currentpen)
{
  for(int i=0; i < g.length; ++i) draw(f,g[i],p);
}

void filldraw(frame f, path[] g, pen fillpen=currentpen,
              pen drawpen=currentpen)
{
  begingroup(f);
  fill(f,g,fillpen);
  draw(f,g,drawpen);
  endgroup(f);
}

path[] complement(frame f, path[] g)
{
  static pair margin=(0.5,0.5);
  return box(minbound(min(f),min(g))-margin,maxbound(max(f),max(g))+margin)^^g;
}

void unfill(frame f, path[] g, bool copy=true)
{
  clip(f,complement(f,g),evenodd,copy);
}

void filloutside(frame f, path[] g, pen p=currentpen, bool copy=true)
{
  fill(f,complement(f,g),p+evenodd,copy);
}

struct filltype 
{
  typedef void fill2(frame f, path[] g, pen fillpen);
  fill2 fill2;
  pen fillpen;
  pen drawpen;

  int type;
  static int Fill=1;
  static int FillDraw=2;
  static int Draw=3;
  static int NoFill=4;
  static int UnFill=5;
      
  void operator init(int type=0, pen fillpen=nullpen, pen drawpen=nullpen,
                     fill2 fill2) {
    this.type=type;
    this.fillpen=fillpen;
    this.drawpen=drawpen;
    this.fill2=fill2;
  }
  void fill(frame f, path[] g, pen p) {fill2(f,g,p);}
}

path[] margin(path[] g, real xmargin, real ymargin) 
{
  if(xmargin != 0 || ymargin != 0) {
    pair M=max(g);
    pair m=min(g);
    real width=M.x-m.x;
    real height=M.y-m.y;
    real xfactor=width > 0 ? (width+2xmargin)/width : 1;
    real yfactor=height > 0 ? (height+2ymargin)/height : 1;
    g=scale(xfactor,yfactor)*g;
    g=shift(0.5*(M+m)-0.5*(max(g)+min(g)))*g;
  }   
  return g;
}

filltype Fill(real xmargin=0, real ymargin=xmargin, pen p=nullpen)
{
  return filltype(filltype.Fill,p,new void(frame f, path[] g, pen fillpen) {
      if(p != nullpen) fillpen=p;
      if(fillpen == nullpen) fillpen=currentpen;
      fill(f,margin(g,xmargin,ymargin),fillpen);
    });
}

filltype FillDraw(real xmargin=0, real ymargin=xmargin,
		  pen fillpen=nullpen, pen drawpen=nullpen)
{
  return filltype(filltype.FillDraw,fillpen,drawpen,
                  new void(frame f, path[] g, pen Drawpen) {
    if(drawpen != nullpen) Drawpen=drawpen;
    pen Fillpen=fillpen == nullpen ? Drawpen : fillpen;
    if(Fillpen == nullpen) Fillpen=currentpen;
    if(Drawpen == nullpen) Drawpen=Fillpen;
     if(cyclic(g[0]))
      filldraw(f,margin(g,xmargin,ymargin),Fillpen,Drawpen);
    else
      draw(f,margin(g,xmargin,ymargin),Drawpen);
    });
}

filltype Draw(real xmargin=0, real ymargin=xmargin, pen p=nullpen)
{
  return filltype(filltype.Draw,drawpen=p,
                  new void(frame f, path[] g, pen drawpen) {
    pen drawpen=p == nullpen ? drawpen : p;
    if(drawpen == nullpen) drawpen=currentpen;
    draw(f,margin(g,xmargin,ymargin),drawpen);
    });
}

filltype NoFill=filltype(filltype.NoFill,new void(frame f, path[] g, pen p) {
    draw(f,g,p);
  });


filltype UnFill(real xmargin=0, real ymargin=xmargin)
{
  return filltype(filltype.UnFill,new void(frame f, path[] g, pen) {
    unfill(f,margin(g,xmargin,ymargin));
    });
}

filltype FillDraw=FillDraw(), Fill=Fill(), Draw=Draw(), UnFill=UnFill(); 

// Fill varying radially from penc at the center of the bounding box to
// penr at the edge.
filltype RadialShade(pen penc, pen penr)
{
  return filltype(new void(frame f, path[] g, pen) {
    pair c=(min(g)+max(g))/2;
    radialshade(f,g,penc,c,0,penr,c,abs(max(g)-min(g))/2);
    });
}

filltype RadialShadeDraw(real xmargin=0, real ymargin=xmargin,
                         pen penc, pen penr, pen drawpen=nullpen)
{
  return filltype(new void(frame f, path[] g, pen Drawpen) {
    if(drawpen != nullpen) Drawpen=drawpen;
    if(Drawpen == nullpen) Drawpen=penc;
    pair c=(min(g)+max(g))/2;
    if(cyclic(g[0]))
      radialshade(f,margin(g,xmargin,ymargin),penc,c,0,penr,c,
                  abs(max(g)-min(g))/2);
    draw(f,margin(g,xmargin,ymargin),Drawpen);
   });
}

// Fill the region in frame dest underneath frame src and return the
// boundary of src.
path fill(frame dest, frame src, filltype filltype=NoFill,
          real xmargin=0, real ymargin=xmargin)
{
  pair z=(xmargin,ymargin);
  path g=box(min(src)-z,max(src)+z);
  filltype.fill(dest,g,nullpen);
  return g;
}

// Add frame dest to frame src with optional grouping and background fill.
void add(frame dest, frame src, bool group, filltype filltype=NoFill,
         bool above=true)
{
  if(above) {
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

void add(frame dest, frame src, filltype filltype,
	 bool above=filltype.type != filltype.UnFill)
{
  if(filltype != NoFill) fill(dest,src,filltype);
  (above ? add : prepend)(dest,src);
}
