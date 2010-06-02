void render(path3 s, real granularity=tubegranularity, void f(path3, real))
{
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

surface bispline(real[][] z, real[][] p, real[][] q, real[][] r,
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
      for(int j=0; j < m; ++j)
        if(condi[j]) ++count;
    }
  }

  surface s=surface(count);
  s.index=new int[n][m];
  int k=-1;
  for(int i=0; i < n; ++i) {
    bool[] condi=all ? null : cond[i];
    real xi=x[i];
    real[] zi=z[i];
    real[] zp=z[i+1];
    real[] ri=r[i];
    real[] rp=r[i+1];
    real[] pi=p[i];
    real[] pp=p[i+1];
    real[] qi=q[i];
    real[] qp=q[i+1];
    real xp=x[i+1];
    real hx=(xp-xi)/3;
    int[] indexi=s.index[i];
    for(int j=0; j < m; ++j) {
      real yj=y[j];
      real yp=y[j+1];
      if(all || condi[j]) {
        triple[][] P=array(4,array(4,O));
        real hy=(yp-yj)/3;
        real hxy=hx*hy;
        // x and y directions
        for(int k=0; k < 4; ++k) {
          P[0][k] += xi*X;
          P[k][0] += yj*Y;
          P[1][k] += (xp+2*xi)/3*X;
          P[k][1] += (yp+2*yj)/3*Y;
          P[2][k] += (2*xp+xi)/3*X;
          P[k][2] += (2*yp+yj)/3*Y;
          P[3][k] += xp*X;
          P[k][3] += yp*Y;
        }
        // z: value 
        P[0][0] += zi[j]*Z;
        P[3][0] += zp[j]*Z;
        P[0][3] += zi[j+1]*Z;
        P[3][3] += zp[j+1]*Z;
        // z: first derivative
        P[1][0] += (P[0][0].z+hx*pi[j])*Z;
        P[1][3] += (P[0][3].z+hx*pi[j+1])*Z;
        P[2][0] += (P[3][0].z-hx*pp[j])*Z;
        P[2][3] += (P[3][3].z-hx*pp[j+1])*Z;
        P[0][1] += (P[0][0].z+hy*qi[j])*Z;
        P[3][1] += (P[3][0].z+hy*qp[j])*Z;
        P[0][2] += (P[0][3].z-hy*qi[j+1])*Z;
        P[3][2] += (P[3][3].z-hy*qp[j+1])*Z;
        // z: second derivative
        P[1][1] += (P[0][1].z+P[1][0].z-P[0][0].z+hxy*ri[j])*Z;
        P[1][2] += (P[0][2].z+P[1][3].z-P[0][3].z-hxy*ri[j+1])*Z;
        P[2][1] += (P[2][0].z+P[3][1].z-P[3][0].z-hxy*rp[j])*Z;
        P[2][2] += (P[2][3].z+P[3][2].z-P[3][3].z+hxy*rp[j+1])*Z;
        s.s[++k]=patch(P);
        indexi[j]=k;
      }
    }
  }
  
  return s;
}

// return the surface described by a real matrix f, interpolated with
// xsplinetype and ysplinetype.
surface surface(real[][] f, real[] x, real[] y,
                splinetype xsplinetype=null, splinetype ysplinetype=xsplinetype,
                bool[][] cond={})
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
  surface s=bispline(f,p,q,r,x,y,cond);
  if(xsplinetype == periodic) s.ucyclic(true);
  if(ysplinetype == periodic) s.vcyclic(true);
  return s;
}

bool uperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length;
  real[] a0=a[0];
  real[] a1=a[n-1];
  real epsilon=sqrtEpsilon*norm(a);
  for(int j=0; j < m; ++j)
    if(abs(a0[j]-a1[j]) > epsilon) return false;
  return true;
}
bool vperiodic(real[][] a) {
  int n=a.length;
  if(n == 0) return false;
  int m=a[0].length-1;
  real epsilon=sqrtEpsilon*norm(a);
  for(int i=0; i < n; ++i)
    if(abs(a[i][0]-a[i][m]) > epsilon) return false;
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
      triple f=(all || (activei[j]=cond(z))) ? f(z) : O;
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
  
  surface sx=surface(fx,ipt,jpt,usplinetype[0],vsplinetype[0],active);
  surface sy=surface(fy,ipt,jpt,usplinetype[1],vsplinetype[1],active);
  surface sz=surface(fz,ipt,jpt,usplinetype[2],vsplinetype[2],active);

  surface s=surface(sx.s.length);
  s.index=new int[nu][nv];
  int k=-1;
  for(int i=0; i < nu; ++i) {
    int[] indexi=s.index[i];
    for(int j=0; j < nv; ++j)
      indexi[j]=++k;
  }

  for(int k=0; k < sx.s.length; ++k) {
    triple[][] Q=new triple[4][];
    for(int i=0; i < 4 ; ++i)
      Q[i]=sequence(new triple(int j) {
          return (sx.s[k].P[i][j].z,sy.s[k].P[i][j].z,sz.s[k].P[i][j].z);
        },4);
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
  return path3(sequence(new triple(int i) {return interp(precontrol(a,i),
                                                         precontrol(b,i),t);},n),
    sequence(new triple(int i) {return interp(point(a,i),point(b,i),t);},n),
    sequence(new triple(int i) {return interp(postcontrol(a,i),
                                              postcontrol(b,i),t);},n),
    sequence(new bool(int i) {return straight(a,i) && straight(b,i);},n),
    cyclic(a) && cyclic(b));
}

struct tube
{
  surface s;
  path3 center;

  void Null(transform3) {}
  void Null(transform3, bool) {}
  
  void operator init(path3 p, real width, int sectors=4,
                     real granularity=tubegranularity,
                     void cylinder(transform3)=Null,
                     void sphere(transform3, bool)=Null,
                     void tube(path3, path3)=null) {
    sectors += sectors % 2; // Must be even.
    int h=quotient(sectors,2);
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
          render(subpath(p,i,i+1),granularity,
                 new void(path3 g, real s) {
                   G=G&g;
                   T.push(i+s);
                 });
        T.push(n);
        T.cyclic=cyclic(p);
        rmf[] rmf=rmf(p,T);
        triple f(pair t) {
          rmf R=rmf[round(t.x)];
          return point(G,t.x)+r*(R.r*cos(t.y)-R.s*sin(t.y));
        }

        real[] v=uniform(0,2pi,sectors);
        static splinetype[] Monotonic={monotonic,monotonic,monotonic};
        static splinetype[] Periodic={periodic,periodic,periodic};
        if(T.length > 0) {
          surface S=surface(f,sequence(T.length),v,Monotonic,Periodic);
          s.append(S);

          // Compute center of tube:
          int n=S.index.length;
          if(T.cyclic) --n;
          triple[] pre=new triple[n+1];
          triple[] point=new triple[n+1];
          triple[] post=new triple[n+1];
          int[] index=S.index[0];
          pre[0]=point[0]=0.5*(S.s[index[0]].P[0][0]+S.s[index[h]].P[0][0]);
          for(int i=0; i < n; ++i) {
            index=S.index[i];
            triple [][] P=S.s[index[0]].P;
            triple [][] Q=S.s[index[h]].P;
            post[i]=0.5*(P[1][0]+Q[1][0]);
            pre[i+1]=0.5*(P[2][0]+Q[2][0]);
            point[i+1]=0.5*(P[3][0]+Q[3][0]);
          }
          index=S.index[n-1];
          post[n]=0.5*(S.s[index[0]].P[3][0]+S.s[index[h]].P[3][0]);
          path3 Center=path3(pre,point,post,array(n+1,false),T.cyclic);
          center=center&Center;

          if(tube != null) {
            triple[] pre=new triple[n+1];
            triple[] point=new triple[n+1];
            triple[] post=new triple[n+1];
            int[] index=S.index[0];
            pre[0]=point[0]=S.s[index[0]].P[0][0];
            for(int i=0; i < n; ++i) {
              index=S.index[i];
              triple [][] P=S.s[index[0]].P;
              post[i]=P[1][0];
              pre[i+1]=P[2][0];
              point[i+1]=P[3][0];
            }
            index=S.index[n-1];
            post[n]=S.s[index[0]].P[3][0];
            tube(Center,path3(pre,point,post,array(n+1,false),T.cyclic));
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
       transform3 t=shift(point(p,i))*t*align(dir(p,i,-1));
       s.append(t*unithemisphere);
       sphere(t,false);
       begin=i;
     }
    generate(subpath(p,begin,n));
  }
}
