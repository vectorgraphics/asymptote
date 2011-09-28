path nullpath;

typedef guide interpolate(... guide[]);

// These numbers identify the side of a specifier in an operator spec or
// operator curl expression:
//  a{out} .. {in}b
restricted int JOIN_OUT=0;
restricted int JOIN_IN=1;

// Define a.. tension t ..b to be equivalent to
//        a.. tension t and t ..b
// and likewise with controls.
tensionSpecifier operator tension(real t, bool atLeast)
{
  return operator tension(t,t,atLeast);
}

guide operator controls(pair z)
{
  return operator controls(z,z);
}

guide[] operator cast(pair[] z)
{
  return sequence(new guide(int i) {return z[i];},z.length);
}

path[] operator cast(pair[] z)
{
  return sequence(new path(int i) {return z[i];},z.length);
}

path[] operator cast(guide[] g)
{
  return sequence(new path(int i) {return g[i];},g.length);
}

guide[] operator cast(path[] g)
{
  return sequence(new guide(int i) {return g[i];},g.length);
}

path[] operator cast(path p)
{
  return new path[] {p};
}

path[] operator cast(guide g)
{
  return new path[] {(path) g};
}

path[] operator ^^ (path p, path q) 
{
  return new path[] {p,q};
}

path[] operator ^^ (path p, explicit path[] q) 
{
  return concat(new path[] {p},q);
}

path[] operator ^^ (explicit path[] p, path q) 
{
  return concat(p,new path[] {q});
}

path[] operator ^^ (explicit path[] p, explicit path[] q) 
{
  return concat(p,q);
}

path[] operator * (transform t, explicit path[] p) 
{
  return sequence(new path(int i) {return t*p[i];},p.length);
}

pair[] operator * (transform t, pair[] z) 
{
  return sequence(new pair(int i) {return t*z[i];},z.length);
}

void write(file file, string s="", explicit path[] x, suffix suffix=none)
{
  write(file,s);
  if(x.length > 0) write(file,x[0]);
  for(int i=1; i < x.length; ++i) {
    write(file,endl);
    write(file," ^^");
    write(file,x[i]);
  }
  write(file,suffix);
}

void write(string s="", explicit path[] x, suffix suffix=endl) 
{
  write(stdout,s,x,suffix);
}

void write(file file, string s="", explicit guide[] x, suffix suffix=none)
{
  write(file,s);
  if(x.length > 0) write(file,x[0]);
  for(int i=1; i < x.length; ++i) {
    write(file,endl);
    write(file," ^^");
    write(file,x[i]);
  }
  write(file,suffix);
}

void write(string s="", explicit guide[] x, suffix suffix=endl) 
{
  write(stdout,s,x,suffix);
}

private string nopoints="nullpath has no points";

pair min(explicit path[] p)
{
  if(p.length == 0) abort(nopoints);
  pair minp=min(p[0]);
  for(int i=1; i < p.length; ++i)
    minp=minbound(minp,min(p[i]));
  return minp;
}

pair max(explicit path[] p)
{
  if(p.length == 0) abort(nopoints);
  pair maxp=max(p[0]);
  for(int i=1; i < p.length; ++i)
    maxp=maxbound(maxp,max(p[i]));
  return maxp;
}

interpolate operator ..(tensionSpecifier t)
{
  return new guide(... guide[] a) {
    if(a.length == 0) return nullpath;
    guide g=a[0];
    for(int i=1; i < a.length; ++i)
      g=g..t..a[i];
    return g;
  };
}

interpolate operator ::=operator ..(operator tension(1,true));
interpolate operator ---=operator ..(operator tension(infinity,true));

// return an arbitrary intersection point of paths p and q
pair intersectionpoint(path p, path q, real fuzz=-1)
{
  real[] t=intersect(p,q,fuzz);
  if(t.length == 0) abort("paths do not intersect");
  return point(p,t[0]);
}

// return an array containing all intersection points of the paths p and q
pair[] intersectionpoints(path p, path q, real fuzz=-1)
{
  real[][] t=intersections(p,q,fuzz);
  return sequence(new pair(int i) {return point(p,t[i][0]);},t.length);
}

pair[] intersectionpoints(explicit path[] p, explicit path[] q, real fuzz=-1)
{
  pair[] z;
  for(int i=0; i < p.length; ++i)
    for(int j=0; j < q.length; ++j)
      z.append(intersectionpoints(p[i],q[j],fuzz));
  return z;
}

struct slice {
  path before,after;
}
  
slice cut(path p, path knife, int n) 
{
  slice s;
  real[][] T=intersections(p,knife);
  if(T.length == 0) {s.before=p; s.after=nullpath; return s;}
  T.cyclic=true;
  real t=T[n][0];
  s.before=subpath(p,0,t);
  s.after=subpath(p,t,length(p));
  return s;
}

slice firstcut(path p, path knife) 
{
  return cut(p,knife,0);
}

slice lastcut(path p, path knife)
{
  return cut(p,knife,-1);
}

pair dir(path p)
{
  return dir(p,length(p));
}

pair dir(path p, path q)
{
  return unit(dir(p)+dir(q));
}

// return the point on path p at arclength L
pair arcpoint(path p, real L)
{
  return point(p,arctime(p,L));
}

// return the direction on path p at arclength L
pair arcdir(path p, real L)
{
  return dir(p,arctime(p,L));
}

// return the time on path p at the relative fraction l of its arclength
real reltime(path p, real l)
{
  return arctime(p,l*arclength(p));
}

// return the point on path p at the relative fraction l of its arclength
pair relpoint(path p, real l)
{
  return point(p,reltime(p,l));
}

// return the direction of path p at the relative fraction l of its arclength
pair reldir(path p, real l)
{
  return dir(p,reltime(p,l));
}

// return the initial point of path p
pair beginpoint(path p)
{
  return point(p,0);
}

// return the point on path p at half of its arclength
pair midpoint(path p)
{
  return relpoint(p,0.5);
}

// return the final point of path p
pair endpoint(path p)
{
  return point(p,length(p));
}

path operator &(path p, cycleToken tok)
{
  int n=length(p);
  if(n < 0) return nullpath;
  if(n == 0) return p--cycle;
  if(cyclic(p)) return p;
  return straight(p,n-1) ? subpath(p,0,n-1)--cycle :
    subpath(p,0,n-1)..controls postcontrol(p,n-1) and precontrol(p,n)..cycle;
}

// return a cyclic path enclosing a region bounded by a list of two or more
// consecutively intersecting paths
path buildcycle(... path[] p)
{
  int n=p.length;
  if(n < 2) return nullpath;
  real[] ta=new real[n];
  real[] tb=new real[n];
  if(n == 2) {
    real[][] t=intersections(p[0],p[1]);
    if(t.length < 2)
      return nullpath;
    int k=t.length-1;
    ta[0]=t[0][0]; tb[0]=t[k][0];
    ta[1]=t[k][1]; tb[1]=t[0][1];
  } else {
    int j=n-1;
    for(int i=0; i < n; ++i) {
      real[][] t=intersections(p[i],p[j]);
      if(t.length == 0)
        return nullpath;
      ta[i]=t[0][0]; tb[j]=t[0][1];
      j=i;
    }
  }

  pair c;
  for(int i=0; i < n ; ++i)
    c += point(p[i],ta[i]);
  c /= n;

  path G;
  for(int i=0; i < n ; ++i) {
    real Ta=ta[i];
    real Tb=tb[i];
    if(cyclic(p[i])) {
      int L=length(p[i]);
      real t=Tb-L;
      if(abs(c-point(p[i],0.5(Ta+t))) <
         abs(c-point(p[i],0.5(Ta+Tb)))) Tb=t;
      while(Tb < Ta) Tb += L;
    }
    G=G&subpath(p[i],Ta,Tb);
  }
  return G&cycle;
}

// return 1 if p strictly contains q,
//       -1 if q strictly contains p,
//        0 otherwise.
int inside(path p, path q, pen fillrule=currentpen)
{
  if(intersect(p,q).length > 0) return 0;
  if(cyclic(p) && inside(p,point(q,0),fillrule)) return 1;
  if(cyclic(q) && inside(q,point(p,0),fillrule)) return -1;
  return 0;
}

// Return an arbitrary point strictly inside a cyclic path p according to
// the specified fill rule.
pair inside(path p, pen fillrule=currentpen)
{
  if(!cyclic(p)) abort("path is not cyclic");
  int n=length(p);
  for(int i=0; i < n; ++i) {
    pair z=point(p,i);
    pair dir=dir(p,i);
    if(dir == 0) continue;
    real[] T=intersections(p,z,z+I*dir);
    // Check midpoints of line segments formed between the
    // corresponding intersection points and z.
    for(int j=0; j < T.length; ++j) {
      if(T[j] != i) {
        pair w=point(p,T[j]);
        pair m=0.5*(z+w);
        if(interior(windingnumber(p,m),fillrule)) return m;
      }
    }
  }
  // cannot find an interior point: path is degenerate
  return point(p,0);
}

// Return all intersection times of path g with the vertical line through (x,0).
real[] times(path p, real x)
{
  return intersections(p,(x,0),(x,1));
}

// Return all intersection times of path g with the horizontal line through
// (0,z.y).
real[] times(path p, explicit pair z)
{
  return intersections(p,(0,z.y),(1,z.y));
}

path randompath(int n, bool cumulate=true, interpolate join=operator ..)
{
  guide g;
  pair w;
  for(int i=0; i <= n; ++i) {
    pair z=(unitrand()-0.5,unitrand()-0.5);
    if(cumulate) w += z; 
    else w=z;
    g=join(g,w);
  }
  return g;
}

path[] strokepath(path g, pen p=currentpen)
{
  path[] G=_strokepath(g,p);
  if(G.length == 0) return G;
  pair center(path g) {return 0.5*(min(g)+max(g));}
  pair center(path[] g) {return 0.5*(min(g)+max(g));}
  return shift(center(g)-center(G))*G;
}
