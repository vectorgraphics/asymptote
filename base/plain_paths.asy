path nullpath;

// Define a.. tension t ..b to be equivalent to
//        a.. tension t and t ..b
// and likewise with controls.
guide operator tension(real t, bool atLeast)
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

path[] operator cast(path g)
{
  return new path[] {g};
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

pair min(explicit path[] g)
{
  pair ming=Infinity;
  for(int i=0; i < g.length; ++i)
    ming=minbound(ming,min(g[i]));
  return ming;
}

pair max(explicit path[] g)
{
  pair maxg=(-infinity,-infinity);
  for(int i=0; i < g.length; ++i)
    maxg=maxbound(maxg,max(g[i]));
  return maxg;
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

pair intersectionpoint(path p, path q, real fuzz=0)
{
  return point(p,intersect(p,q,fuzz).x);
}

struct slice {
  path before,after;
}
  
slice operator init() {return new slice;}

slice firstcut(path g, path knife) 
{
  slice s;
  real t=intersect(g,knife).x;
  if (t < 0) {s.before=g; s.after=nullpath; return s;}
  s.before=subpath(g,0,min(t,intersect(g,reverse(knife)).x));
  s.after=subpath(g,min(t,intersect(g,reverse(knife)).x),length(g));
  return s;
}

slice lastcut(path g, path knife) 
{
  slice s=firstcut(reverse(g),knife);
  path before=reverse(s.after);
  s.after=reverse(s.before);
  s.before=before;
  return s;
}

pair dir(path g)
{
  return dir(g,length(g));
}

pair dir(path g, path h)
{
  return 0.5*(dir(g)+dir(h));
}

// return the point on path g at arclength L
pair arcpoint(path g, real L)
{
  return point(g,arctime(g,L));
}

// return the direction on path g at arclength L
pair arcdir(path g, real L)
{
  return dir(g,arctime(g,L));
}

// return the time on path g at the given relative fraction of its arclength
real reltime(path g, real fraction)
{
  return arctime(g,fraction*arclength(g));
}

// return the point on path g at the given relative fraction of its arclength
pair relpoint(path g, real l)
{
  return point(g,reltime(g,l));
}

// return the direction of path g at the given relative fraction of its
// arclength
pair reldir(path g, real l)
{
  return dir(g,reltime(g,l));
}

// return the initial point of path g
pair beginpoint(path g)
{
  return point(g,0);
}

// return the point on path g at half of its arclength
pair midpoint(path g)
{
  return relpoint(g,0.5);
}

// return the final point of path g
pair endpoint(path g)
{
  return point(g,length(g));
}

// return the path surrounding a region bounded by a list of consecutively
// intersecting paths
path buildcycle(... path[] g)
{
  int n=g.length;
  real[] ta=new real[n];
  real[] tb=new real[n];
  int j=n-1;
  for(int i=0; i < n; ++i) {
    pair t=intersect(g[i],reverse(g[j]));
    if(t == (-1,-1))
      abort("Paths "+(string) i+" and " +(string) j+" do not intersect");
    ta[i]=t.x; tb[j]=length(g[j])-t.y;
    j=i;
  }
  path G;
  for(int i=0; i < n ; ++i) 
    G=G..subpath(g[i],ta[i],tb[i]);
  return G..cycle;
}
