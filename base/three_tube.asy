struct rmf {
  triple p,r,t,s;
  void operator init(triple p, triple r, triple t) {
    this.p=p;
    this.r=r;
    this.t=t;
    s=cross(t,r);
  }

  transform3 transform() {
    return transform3(r,s,t);
  }
}

// Rotation minimizing frame
// http://www.cs.hku.hk/research/techreps/document/TR-2007-07.pdf
rmf[] rmf(path3 g, real[] t, triple perp=O)
{
  triple T=dir(g,0);
  triple Tp=abs(perp) < sqrtEpsilon ? perp(T) : unit(perp);
  rmf[] R=new rmf[t.length];
  R[0]=rmf(point(g,0),Tp,T);
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

rmf[] rmf(triple z0, triple c0, triple c1, triple z1, real[] t, triple perp=O)
{
  static triple s0;

  real norm=sqrtEpsilon*max(abs(z0),abs(c0),abs(c1),abs(z1));

  // Special case of dir for t in (0,1].
  triple dir(real t) {
    if(t == 1) {
      triple dir=z1-c1;
      if(abs(dir) > norm) return unit(dir);
      dir=2.0*c1-c0-z1;
      if(abs(dir) > norm) return unit(dir);
      return unit(z1-z0+3.0*(c0-c1));
    }
    triple a=z1-z0+3.0*(c0-c1);
    triple b=2.0*(z0+c1)-4.0*c0;
    triple c=c0-z0;
    triple dir=a*t*t+b*t+c;
    if(abs(dir) > norm) return unit(dir);
    dir=2.0*a*t+b;
    if(abs(dir) > norm) return unit(dir);
    return unit(a);
  }

  triple T=c0-z0;
  if(abs(T) < norm) {
    T=z0-2*c0+c1;
    if(abs(T) < norm)
      T=z1-z0+3.0*(c0-c1);
  }
  T=unit(T);
  triple Tp=perp == O ? cross(s0,T) : perp;
  Tp=abs(Tp) < sqrtEpsilon ? perp(T) : unit(Tp);
  rmf[] R=new rmf[t.length];
  R[0]=rmf(z0,Tp,T);

  for(int i=1; i < t.length; ++i) {
    rmf Ri=R[i-1];
    real t=t[i];
    triple p=bezier(z0,c0,c1,z1,t);
    triple v1=p-Ri.p;
    if(v1 != O) {
      triple r=Ri.r;
      triple u1=unit(v1);
      triple ti=Ri.t;
      triple tp=ti-2*dot(u1,ti)*u1;
      ti=dir(t);
      triple rp=r-2*dot(u1,r)*u1;
      triple u2=unit(ti-tp);
      rp=rp-2*dot(u2,rp)*u2;
      R[i]=rmf(p,unit(rp),unit(ti));
    } else
      R[i]=R[i-1];
  }
  s0=R[t.length-1].s;
  return R;
}

surface tube(triple z0, triple c0, triple c1, triple z1, real w)
{
  surface s;
  static real[] T={0,1/3,2/3,1};
  rmf[] rmf=rmf(z0,c0,c1,z1,T);

  real aw=a*w;
  triple[] arc={(w,0,0),(w,aw,0),(aw,w,0),(0,w,0)};
  triple[] g={z0,c0,c1,z1};

  void f(transform3 R) {
    triple[][] P=new triple[4][];
    for(int i=0; i < 4; ++i) {
      transform3 T=shift(g[i])*rmf[i].transform()*R;
      P[i]=new triple[] {T*arc[0],T*arc[1],T*arc[2],T*arc[3]};
    }
    s.push(patch(P,copy=false));
  }

  f(identity4);
  f(t1);
  f(t2);
  f(t3);

  s.PRCprimitive=false;
  s.draw=new void(frame f, transform3 t=identity4, material[] m,
                  light light=currentlight, render render=defaultrender)
    {
     material m=material(m[0],light);
     drawTube(f,t*g,w,m.p,m.opacity,m.shininess,m.metallic,m.fresnel0,
              t*min(s),t*max(s),m.opacity == 1);
    };
  return s;
}

real tubethreshold=20;

// Note: casting an array of surfaces to a single surface will disable
// primitive compression.
surface operator cast(surface[] s) {
  surface S;
  for(surface p : s)
    S.append(p);
  return S;
}

struct tube
{
  surface[] s;
  path3 center; // tube axis

  void Null(transform3) {}
  void Null(transform3, bool) {}
  
  surface[] render(path3 g, real r) {
    triple z0=point(g,0);
    triple c0=postcontrol(g,0);
    triple c1=precontrol(g,1);
    triple z1=point(g,1);
    real norm=sqrtEpsilon*max(abs(z0),abs(c0),abs(c1),abs(z1));
    surface[] s;
    void Split(triple z0, triple c0, triple c1, triple z1,
               real depth=mantissaBits) {
      if(depth > 0) {
        pair threshold(triple z0, triple c0, triple c1) {
          triple u=c1-z0;
          triple v=c0-z0;
          real x=abs(v);
          return (x,abs(u*x^2-dot(u,v)*v));
        }

        pair a0=threshold(z0,c0,c1);
        pair a1=threshold(z1,c1,c0);
        real rL=r*arclength(z0,c0,c1,z1)*tubethreshold;
        if((a0.x >= norm && rL*a0.y^2 > a0.x^8) || 
           (a1.x >= norm && rL*a1.y^2 > a1.x^8)) {
          triple m0=0.5*(z0+c0);
          triple m1=0.5*(c0+c1);
          triple m2=0.5*(c1+z1);
          triple m3=0.5*(m0+m1);
          triple m4=0.5*(m1+m2);
          triple m5=0.5*(m3+m4);
          --depth;
          Split(z0,m0,m3,m5,depth);
          Split(m5,m4,m2,z1,depth);
          return;
        }
      }

      s.push(tube(z0,c0,c1,z1,r));
    }
    Split(z0,c0,c1,z1);
    return s;
  }

  void operator init(path3 p, real width) {
    center=p;
    real r=0.5*width;

    void generate(path3 p) {
      int n=length(p);
      for(int i=0; i < n; ++i) {
        if(straight(p,i)) {
          triple v=point(p,i);
          triple u=point(p,i+1)-v;
          transform3 t=shift(v)*align(unit(u))*scale(r,r,abs(u));
          // Draw opaque surfaces with core for better small-scale rendering.
          surface unittube=t*unitcylinder;
          unittube.draw=unitcylinderDraw(core=true);
          s.push(unittube);
        } else
          s.append(render(subpath(p,i,i+1),r));
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
        s.push(shift(point(p,i))*T*(straight(p,i-1) && straight(p,i) ?
                                    unithemisphere : unitsphere));
        begin=i;
      }
    path3 g=subpath(p,begin,n);
    generate(g);
  }
}
