void render(path3 s, void f(path3, real), render render=defaultrender)
{
  real granularity=render.tubegranularity;
  void Split(triple z0, triple c0, triple c1, triple z1, real t0=0, real t1=1,
             real depth=mantissaBits) {
    if(depth > 0) {
      real S=straightness(z0,c0,c1,z1);
      if(S > 0) {
        --depth;
        if(S > max(granularity*max(abs(z0),abs(c0),abs(c1),abs(z1)))) {
          triple m0=0.5*(z0+c0);
          triple m1=0.5*(c0+c1);
          triple m2=0.5*(c1+z1);
          triple m3=0.5*(m0+m1);
          triple m4=0.5*(m1+m2);
          triple m5=0.5*(m3+m4);
          real tm=0.5*(t0+t1);
          Split(z0,m0,m3,m5,t0,tm,depth);
          Split(m5,m4,m2,z1,tm,t1,depth);
          return;
        }
      }
    }
    f(z0..controls c0 and c1..z1,t0);
  }
  Split(point(s,0),postcontrol(s,0),precontrol(s,1),point(s,1));
}

struct rmf
{
  triple p,r,t,s;
  void operator init(triple p, triple r, triple t)
  {
    this.p=p;
    this.r=r;
    this.t=t;
    s=cross(t,r);
  }
}

// Rotation minimizing frame
// http://www.cs.hku.hk/research/techreps/document/TR-2007-07.pdf
rmf[] rmf(path3 g, real[] t)
{
  rmf[] R=new rmf[t.length];
  triple d=dir(g,0);
  R[0]=rmf(point(g,0),perp(d),d);
  for(int i=1; i < t.length; ++i) {
    rmf Ri=R[i-1];
    real t=t[i];
    triple p=point(g,t);
    triple v1=p-Ri.p;
    if(v1 != O) {
      triple r=Ri.r;
      triple u1=unit(v1);
      triple ti=Ri.t;
      triple tp=ti-2*dot(u1,ti)*u1;
      ti=dir(g,t);
      triple rp=r-2*dot(u1,r)*u1;
      triple u2=unit(ti-tp);
      rp=rp-2*dot(u2,rp)*u2;
      R[i]=rmf(p,unit(rp),unit(ti));
    } else
      R[i]=R[i-1];
  }
  return R;
}

private real[][][] bispline0(real[][] z, real[][] p, real[][] q, real[][] r,
                             real[] x, real[] y, bool[][] cond={})
{ // z[i][j] is the value at (x[i],y[j])
  // p and q are the first derivatives with respect to x and y, respectively
  // r is the second derivative ddu/dxdy
  int n=x.length-1;
  int m=y.length-1;

  bool all=cond.length == 0;

  int count;
  if(all)
    count=n*m;
  else {
    count=0;
    for(int i=0; i < n; ++i) {
      bool[] condi=cond[i];
      bool[] condp=cond[i+1];
      for(int j=0; j < m; ++j)
        if(all || (condi[j] && condi[j+1] && condp[j] && condp[j+1])) 
          ++count;
    }
  }

  real[][][] s=new real[count][][];
  int k=0;
  for(int i=0; i < n; ++i) {
    int ip=i+1;
    real xi=x[i];
    real xp=x[ip];
    real hx=(xp-xi)/3;
    real[] zi=z[i];
    real[] zp=z[ip];
    real[] ri=r[i];
    real[] rp=r[ip];
    real[] pi=p[i];
    real[] pp=p[ip];
    real[] qi=q[i];
    real[] qp=q[ip];
    bool[] condi=all ? null : cond[i];
    bool[] condp=all ? null : cond[i+1];
    for(int j=0; j < m; ++j) {
      if(all || (condi[j] && condi[j+1] && condp[j] && condp[j+1])) {
        real yj=y[j];
        int jp=j+1;
        real yp=y[jp];
        real hy=(yp-yj)/3;
        real hxy=hx*hy;
        real zij=zi[j];
        real zip=zi[jp];
        real zpj=zp[j];
        real zpp=zp[jp];
        real pij=hx*pi[j];
        real ppj=hx*pp[j];
        real qip=hy*qi[jp];
        real qpp=hy*qp[jp];
        real zippip=zip+hx*pi[jp];
        real zppmppp=zpp-hx*pp[jp];
        real zijqij=zij+hy*qi[j];
        real zpjqpj=zpj+hy*qp[j];
        
        s[k]=new real[][] {{zij,zijqij,zip-qip,zip},
                           {zij+pij,zijqij+pij+hxy*ri[j],
                            zippip-qip-hxy*ri[jp],zippip},
                           {zpj-ppj,zpjqpj-ppj-hxy*rp[j],
                            zppmppp-qpp+hxy*rp[jp],zppmppp},
                           {zpj,zpjqpj,zpp-qpp,zpp}};
        ++k;
      }
    }
  }
  
  return s;
}

// return the surface values described by a real matrix f, interpolated with
// xsplinetype and ysplinetype.
real[][][] bispline(real[][] f, real[] x, real[] y,
                    splinetype xsplinetype=null,
                    splinetype ysplinetype=xsplinetype, bool[][] cond={})
{
  real epsilon=sqrtEpsilon*norm(y);
  if(xsplinetype == null)
    xsplinetype=(abs(x[0]-x[x.length-1]) <= epsilon) ? periodic : notaknot;
  if(ysplinetype == null)
    ysplinetype=(abs(y[0]-y[y.length-1]) <= epsilon) ? periodic : notaknot;
  int n=x.length; int m=y.length;
  real[][] ft=transpose(f);
  real[][] tp=new real[m][];
  for(int j=0; j < m; ++j)
    tp[j]=xsplinetype(x,ft[j]);
  real[][] q=new real[n][];
  for(int i=0; i < n; ++i)
    q[i]=ysplinetype(y,f[i]);
  real[][] qt=transpose(q);
  real[] d1=xsplinetype(x,qt[0]);
  real[] d2=xsplinetype(x,qt[m-1]);
  real[][] r=new real[n][];
  real[][] p=transpose(tp);
  for(int i=0; i < n; ++i)
    r[i]=clamped(d1[i],d2[i])(y,p[i]);
  return bispline0(f,p,q,r,x,y,cond);
}

bool uperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length;
  real[] a0=a[0];
  real[] a1=a[n-1];
  for(int j=0; j < m; ++j) {
    real norm=0;
    for(int i=0; i < n; ++i)
      norm=max(norm,abs(a[i][j]));
    real epsilon=sqrtEpsilon*norm;
    if(abs(a0[j]-a1[j]) > epsilon) return false;
  }
  return true;
}
bool vperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length-1;
  for(int i=0; i < n; ++i) {
    real[] ai=a[i];
    real epsilon=sqrtEpsilon*norm(ai);
    if(abs(ai[0]-ai[m]) > epsilon) return false;
  }
  return true;
}

// return the surface described by a parametric function f evaluated at u and v
// and interpolated with usplinetype and vsplinetype.
surface surface(triple f(pair z), real[] u, real[] v,
                splinetype[] usplinetype, splinetype[] vsplinetype=Spline,
                bool cond(pair z)=null)
{
  int nu=u.length-1;
  int nv=v.length-1;
  real[] ipt=sequence(u.length);
  real[] jpt=sequence(v.length);
  real[][] fx=new real[u.length][v.length];
  real[][] fy=new real[u.length][v.length];
  real[][] fz=new real[u.length][v.length];

  bool[][] active;
  bool all=cond == null;
  if(!all) active=new bool[u.length][v.length];

  for(int i=0; i <= nu; ++i) {
    real ui=u[i];
    real[] fxi=fx[i];
    real[] fyi=fy[i];
    real[] fzi=fz[i];
    bool[] activei=all ? null : active[i];
    for(int j=0; j <= nv; ++j) {
      pair z=(ui,v[j]);
      if(!all) activei[j]=cond(z);
      triple f=f(z);
      fxi[j]=f.x;
      fyi[j]=f.y;
      fzi[j]=f.z;
    }
  }

  if(usplinetype.length == 0) {
    usplinetype=new splinetype[] {uperiodic(fx) ? periodic : notaknot,
                                  uperiodic(fy) ? periodic : notaknot,
                                  uperiodic(fz) ? periodic : notaknot};
  } else if(usplinetype.length != 3) abort("usplinetype must have length 3");

  if(vsplinetype.length == 0) {
    vsplinetype=new splinetype[] {vperiodic(fx) ? periodic : notaknot,
                                  vperiodic(fy) ? periodic : notaknot,
                                  vperiodic(fz) ? periodic : notaknot};
  } else if(vsplinetype.length != 3) abort("vsplinetype must have length 3");
  
  real[][][] sx=bispline(fx,ipt,jpt,usplinetype[0],vsplinetype[0],active);
  real[][][] sy=bispline(fy,ipt,jpt,usplinetype[1],vsplinetype[1],active);
  real[][][] sz=bispline(fz,ipt,jpt,usplinetype[2],vsplinetype[2],active);

  surface s=surface(sx.length);
  s.index=new int[nu][nv];
  int k=-1;
  for(int i=0; i < nu; ++i) {
    int[] indexi=s.index[i];
    for(int j=0; j < nv; ++j)
      indexi[j]=++k;
  }

  for(int k=0; k < sx.length; ++k) {
    triple[][] Q=new triple[4][];
    real[][] Px=sx[k];
    real[][] Py=sy[k];
    real[][] Pz=sz[k];
    for(int i=0; i < 4 ; ++i) {
      real[] Pxi=Px[i];
      real[] Pyi=Py[i];
      real[] Pzi=Pz[i];
      Q[i]=new triple[] {(Pxi[0],Pyi[0],Pzi[0]),
                         (Pxi[1],Pyi[1],Pzi[1]),
                         (Pxi[2],Pyi[2],Pzi[2]),
                         (Pxi[3],Pyi[3],Pzi[3])};
    }
    s.s[k]=patch(Q);
  }

  if(usplinetype[0] == periodic && usplinetype[1] == periodic &&
     usplinetype[1] == periodic) s.ucyclic(true);

  if(vsplinetype[0] == periodic && vsplinetype[1] == periodic &&
     vsplinetype[1] == periodic) s.vcyclic(true);
  
  return s;
}

path3 interp(path3 a, path3 b, real t) 
{
  int n=size(a);
  return path3(sequence(new triple(int i) {
        return interp(precontrol(a,i),precontrol(b,i),t);},n),
    sequence(new triple(int i) {return interp(point(a,i),point(b,i),t);},n),
    sequence(new triple(int i) {return interp(postcontrol(a,i),
                                              postcontrol(b,i),t);},n),
    sequence(new bool(int i) {return straight(a,i) && straight(b,i);},n),
    cyclic(a) && cyclic(b));
}

struct tube
{
  surface s;
  path3 center; // tube axis

  void Null(transform3) {}
  void Null(transform3, bool) {}
  
  void operator init(path3 p, real width, render render=defaultrender,
                     void cylinder(transform3)=Null,
                     void sphere(transform3, bool half)=Null,
                     void pipe(path3, path3)=null) {
    real r=0.5*width;

    void generate(path3 p) {
      int n=length(p);
      if(piecewisestraight(p)) {
        for(int i=0; i < n; ++i) {
          triple v=point(p,i);
          triple u=point(p,i+1)-v;
          transform3 t=shift(v)*align(unit(u))*scale(r,r,abs(u));
          s.append(t*unitcylinder);
          cylinder(t);
        }
        center=center&p;
      } else {
        real[] T;
        path3 G;
        for(int i=0; i < n; ++i)
          render(subpath(p,i,i+1),
                 new void(path3 g, real s) {
                   G=G&g;
                   T.push(i+s);
                 },render);
        T.push(n);
        T.cyclic=cyclic(p);
        rmf[] rmf=rmf(p,T);
        triple f(pair t) {
          rmf R=rmf[round(t.x)];
          int n=round(t.y);
          static real[] x={1,0,-1,0};
          static real[] y={0,1,0,-1};
          return point(G,t.x)+r*(R.r*x[n]-R.s*y[n]);
        }

        static real[] v={0,1,2,3,0};
        static real[] circular(real[] x, real[] y) {
          static real a=8/3*(sqrt(2)-1);
          return a*periodic(x,y);
        }
        
        static splinetype[] Monotonic={monotonic,monotonic,monotonic};
        static splinetype[] Circular={circular,circular,circular};
        if(T.length > 0) {
          surface S=surface(f,sequence(T.length),v,Monotonic,Circular);
          s.append(S);

          // Compute center of tube:
          int n=S.index.length;
          if(T.cyclic) --n;
          triple[] pre=new triple[n+1];
          triple[] point=new triple[n+1];
          triple[] post=new triple[n+1];

          int[] index=S.index[0];
          triple Point;
          for(int m=0; m < 4; ++m)
            Point += S.s[index[m]].P[0][0];
          pre[0]=point[0]=0.25*Point;
            
          for(int i=0; i < n; ++i) {
            index=S.index[i];
            triple Pre,Point,Post;
            for(int m=0; m < 4; ++m) {
              triple [][] P=S.s[index[m]].P;
              Post += P[1][0];
              Pre += P[2][0];
              Point += P[3][0];
            }
            post[i]=0.25*Post;
            pre[i+1]=0.25*Pre;
            point[i+1]=0.25*Point;

          }

          index=S.index[n-1];
          triple Post;
          for(int m=0; m < 4; ++m)
            Post += S.s[index[m]].P[3][0];
          post[n]=0.25*Post;

          bool[] b=array(n+1,false);
          path3 Center=path3(pre,point,post,b,T.cyclic);
          center=center&Center;

          if(pipe != null) { // Compute path along tube
            triple[] pre=new triple[n+1];
            triple[] point=new triple[n+1];
            triple[] post=new triple[n+1];
            pre[0]=point[0]=S.s[S.index[0][0]].P[0][0];
            for(int i=0; i < n; ++i) {
              triple [][] P=S.s[S.index[i][0]].P;
              post[i]=P[1][0];
              pre[i+1]=P[2][0];
              point[i+1]=P[3][0];
            }
            post[n]=S.s[S.index[n-1][0]].P[3][0];
            pipe(Center,path3(pre,point,post,b,T.cyclic));
          }
        }
      }
    }
    
    transform3 t=scale3(r);
    bool cyclic=cyclic(p);
    int begin=0;
    int n=length(p);
    for(int i=cyclic ? 0 : 1; i < n; ++i)
      if(abs(dir(p,i,1)-dir(p,i,-1)) > sqrtEpsilon) {
        generate(subpath(p,begin,i));
        triple dir=dir(p,i,-1);
        transform3 T=t*align(dir);
        s.append(shift(point(p,i))*T*(dir != O ? unithemisphere : unitsphere));
        sphere(shift(point(center,length(center)))*T,
               half=straight(p,i-1) && straight(p,i));
        begin=i;
      }
    generate(subpath(p,begin,n));
  }
}
