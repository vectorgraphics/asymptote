// Author: Philippe Ivaldi
// Based on this paper: 
// http: //www.cs.hku.hk/research/techreps/document/TR-2007-07.pdf
// Note: the additional rotation for a cyclic smooth spine curve is not
// yet properly determined.
// Todo: Implement variational principles for RMF with boundary conditions: 
//       minimum total angular speed OR minimum total squared angular speed

import three;

// A 3D version of roundedpath(path, real).
path3 roundedpath(path3 A, real r) {
  // Author of this routine: Jens Schwaiger
  guide3 rounded;
  triple before, after, indir, outdir;
  int len=length(A);
  bool guideclosed=cyclic(A);
  if(len < 2) {return A;};
  if(guideclosed) {rounded=point(point(A,0)--point(A,1),r);}
  else {rounded=point(A,0);}
  for(int i=1; i < len; i=i+1) {
    before=point(point(A,i)--point(A,i-1),r);
    after=point(point(A,i)--point(A,i+1),r);
    indir=dir(point(A,i-1)--point(A,i),1);
    outdir=dir(point(A,i)--point(A,i+1),1);
    rounded=rounded--before{indir}..{outdir}after;
  }
  if(guideclosed) {
    before=point(point(A,0)--point(A,len-1),r);
    indir=dir(point(A,len-1)--point(A,0),1);
    outdir=dir(point(A,0)--point(A,1),1);
    rounded=rounded--before{indir}..{outdir}cycle;
  } else rounded=rounded--point(A,len);

  return rounded;
}

real degrees(triple u, triple v, triple n=cross(u,v))
{
  return degrees(acos1(dot(unit(u),unit(v))));
}

real[] sample(path3 g, real r, real step=0)
{
  static real epsilon=sqrt(realEpsilon);
  real[] t;
  int n=length(g);
  if(step <= 0) {
    for(int i=0; i < n; ++i) {
      real S=straightness(g,i);
      if(S < epsilon*r) {
	t.push(i);
      } else {
	if(cyclic(g) && i == n-1) {
	  path3 s=subpath(g,i+1,i);
	  real[] tt;
	  real endtime=0;
	  while(endtime < 1) {
	    endtime=takeStep(s,endtime,r);
	    tt.push(i+1-endtime);
	  }
	  for(int j=0; j < tt.length; ++j) t.push(tt[tt.length-1-j]);
	} else {
	  path3 s=subpath(g,i,i+1);
	  real endtime=0;
	  while(endtime < 1) {
	    t.push(i+endtime);
	    endtime=takeStep(s,endtime,r);
	  }
	}
      }
    }
    t.push(n);
  } else {
    int nb=ceil(n/step);
    step=n/nb;
    for(int i=0; i <= nb; ++i)
      t.push(i*step);
  }
  return t;
}

struct Rmf
{
  triple p,r,t; // s=cross(t,r);
  real reltime;
  void operator init(triple p, triple r, triple t, real reltime)
  {
    this.p=p;
    this.r=r;
    this.t=t;
    this.reltime=reltime;
  }
}

private Rmf[] rmf(path3 g, Rmf U0=Rmf(O,O,O,0), real[] t)
{
  static real epsilon=sqrt(realEpsilon);
  bool cyclic=cyclic(g);
  t.cyclic(cyclic);
  if(U0.t == O) {
    triple d=dir(g,0);
    U0=Rmf(point(g,0),perp(d),d,0);
  }
  real l=length(g);
  Rmf[] R={U0};
  triple rp,v1,v2,tp,ti,p;
  real c;

  for(int i=0; i < t.length-1; ++i) {
    p=point(g,t[i+1]);
    v1=p-R[i].p;
    c=dot(v1,v1);
    if(c != 0) {
      rp=R[i].r-2*dot(v1,R[i].r)*v1/c;
      ti=R[i].t;
      tp=ti-2*dot(v1,ti)*v1/c;
      ti=dir(g,t[i+1]);
      v2=ti-tp;
      rp=rp-2*dot(v2,rp)*v2/dot(v2,v2);
      R.push(Rmf(p,unit(rp),unit(ti),t[i+1]/l));
    } else R.push(R[R.length-1]);
  }
  return R;
}

restricted int coloredNodes=1;
restricted int coloredSegments=2;

struct coloredpath
{
  path p;
  pen[] pens(real);
  bool usepens=false;
  int colortype=coloredSegments;

  void operator init(path p, pen[] pens=new pen[] {currentpen},
		     int colortype=coloredSegments)
  {
    this.p=p;
    this.pens=new pen[] (real t) {return pens;};
    this.usepens=true;
    this.colortype=colortype;
  }

  void operator init(path p, pen[] pens(real), int colortype=coloredSegments)
  {
    this.p=p;
    this.pens=pens;
    this.usepens=true;
    this.colortype=colortype;
  }

  void operator init(path p, pen pen(real))
  {
    this.p=p;
    this.pens=new pen[] (real t) {return new pen[] {pen(t)};};
    this.usepens=true;
    this.colortype=coloredSegments;
  }
}

coloredpath operator cast(path p)
{
  coloredpath cp=coloredpath(p);
  cp.usepens=false;
  return cp;
}

coloredpath operator cast(guide p)
{
  return coloredpath(p);
}

private surface surface(Rmf[] R, coloredpath cp,transform T(real)=
			new transform(real t) {return identity();},
			bool cyclic=false)
{
  path g=cp.p;
  int l=length(g);
  bool[] planar;
  for(int i=0; i < l; ++i)
    planar[i]=straight(g,i);

  surface s;
  path3 sec=path3(T(R[0].reltime)*g);
  real adjust=0;
  if(cyclic)
    adjust=degrees(R[0].r,sgn(dot(cross(R[0].t,R[0].r),
				  cross(R[R.length-1].t,R[R.length-1].r)))
		   *R[R.length-1].r)/R.length;
  path3 sec1=shift(R[0].p)*transform3(R[0].r,cross(R[0].t,R[0].r),R[0].t)*sec,
    sec2;

  for(int i=1; i < R.length; ++i) {
    sec=path3(T(R[i].reltime)*g);
    sec2=shift(R[i].p)*transform3(R[i].r,cross(R[i].t,R[i].r),R[i].t)*
      rotate(i*adjust,Z)*sec;
    for(int j=0; j < l; ++j) {
      surface st=surface(subpath(sec1,j,j+1)--subpath(sec2,j+1,j)--cycle,
			 planar=planar[j]);
      if(cp.usepens) {
        pen[] tp1=cp.pens(R[i-1].reltime), tp2=cp.pens(R[i].reltime);
        tp1.cyclic(true); tp2.cyclic(true);
        if(cp.colortype == coloredSegments) {
          st.colors(new pen[][] {{tp1[j],tp1[j],tp2[j],tp2[j]}});
	} else {
          st.colors(new pen[][] {{tp1[j],tp1[j+1],tp2[j+1],tp2[j]}});
	}
      }
      s.append(st);
    }
    sec1=sec2;
  }
  return s;
}

surface tube(path3 g, coloredpath section,
             transform T(real)=new transform(real t) {return identity();},
             real corner=1, real step=0)
{
  pair M=max(section.p), m=min(section.p);
  real[] t=sample(g,max(M.x-m.x,M.y-m.y)/max(realEpsilon,abs(corner)),step);
  return surface(rmf(g,t),section,T,cyclic(g));
}
