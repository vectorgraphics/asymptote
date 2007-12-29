path nullpath;

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
  guide[] g=new guide[z.length];
  for(int i=0; i < z.length; ++i) g[i]=z[i];
  return g;
}

path[] operator cast(pair[] z)
{
  path[] g=new path[z.length];
  for(int i=0; i < z.length; ++i) g[i]=z[i];
  return g;
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
  path[] P;
  for(int i=0; i < p.length; ++i) P[i]=t*p[i];
  return P;
}

pair[] operator * (transform t, pair[] z) 
{
  pair[] Z;
  for(int i=0; i < z.length; ++i) Z[i]=t*z[i];
  return Z;
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

pair min(explicit path[] p)
{
  pair minp=Infinity;
  for(int i=0; i < p.length; ++i)
    minp=minbound(minp,min(p[i]));
  return minp;
}

pair max(explicit path[] p)
{
  pair maxp=(-infinity,-infinity);
  for(int i=0; i < p.length; ++i)
    maxp=maxbound(maxp,max(p[i]));
  return maxp;
}

guide operator ::(... guide[] a)
{
  if(a.length == 0) return nullpath;
  guide g=a[0];
  for(int i=1; i < a.length; ++i)
    g=g..operator tension(1,true)..a[i];
  return g;
}

guide operator ---(... guide[] a)
{
  if(a.length == 0) return nullpath;
  guide g=a[0];
  for(int i=1; i < a.length; ++i)
    g=g..operator tension(infinity,true)..a[i];
  return g;
}

// return an arbitrary intersection point of paths p and q
pair intersectionpoint(path p, path q, real fuzz=0)
{
  real[] t=intersect(p,q,fuzz);
  if(t.length == 0) abort("paths do not intersect");
  return point(p,t[0]);
}

// return an array containing all intersection points of the paths p and q
pair[] intersectionpoints(path p, path q, real fuzz=0)
{
  real[][] t=intersections(p,q,fuzz);
  pair[] z=new pair[t.length];
  for(int i=0; i < t.length; ++i)
    z[i]=point(p,t[i][0]);
  return z;
}

pair[] intersectionpoints(explicit path[] p, explicit path[] q, real fuzz=0)
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
  T.cyclic(true);
  if(T.length == 0) {s.before=p; s.after=nullpath; return s;}
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

pair dir(path p, path h)
{
  return 0.5*(dir(p)+dir(h));
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

// return the path surrounding a region bounded by a list of consecutively
// intersecting paths
path buildcycle(... path[] p)
{
  int n=p.length;
  real[] ta=new real[n];
  real[] tb=new real[n];
  int j=n-1;
  for(int i=0; i < n; ++i) {
    real[] t=intersect(p[i],reverse(p[j]));
    if(t.length == 0)
      abort("Paths "+(string) i+" and " +(string) j+" do not intersect");
    ta[i]=t[0]; tb[j]=length(p[j])-t[1];
    j=i;
  }
  path G;
  for(int i=0; i < n ; ++i) 
    G=G..subpath(p[i],ta[i],tb[i]);
  return G..cycle;
}

path operator &(path p, cycleToken tok)
{
  int n=length(p);
  if(n < 0) return nullpath;
  pair a=point(p,0);
  pair b=point(p,n);
  static real Fuzz=10.0*realEpsilon;
  if(abs(a-b) > Fuzz*max(abs(a),abs(b)))
    abort("paths in concatenation do not meet");
  return subpath(p,0,n-1)..controls postcontrol(p,n-1) and precontrol(p,n)..
    cycle;
}

