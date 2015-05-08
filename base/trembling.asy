// Copyright(c) 2008, Philippe Ivaldi.
// Simplified by John Bowman 02Feb2011
// http: //www.piprime.fr/
// trembling.asy: handwriting package for the software Asymptote.

// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
//(at your option) any later version.

// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

// COMMENTARY: 

// THANKS: 

// BUGS: 
// magnetic points are experimental...

// CODE: 

real magneticRadius=1; // unit is bp in postscript coordinates.
real trembleFuzz(){return min(1e-3,magneticRadius/10);}

real trembleAngle=4, trembleFrequency=0.5, trembleRandom=2;

struct tremble
{
  static real test=5;
  
  real angle,frequency,random,fuzz;

  pair[] single(pair[] P)
  {
    pair[] op;
    bool allow;
    for(int i=0; i < P.length-1; ++i) {
      allow=true;
      for(int j=i+1; j < P.length; ++j) {
        if(abs(P[i]-P[j]) < magneticRadius) {
          allow=false;
          break;
        }
      }
      if(allow) op.push(P[i]);
    }
    if(P.length > 0) op.push(P[P.length-1]);
    return op;
  }

  real atime(pair m, path g, real fuzz=trembleFuzz())
  {// Return the time of the point on path g nearest to m, within fuzz.
    if(length(g) == 0) return 0.0;
    real[] t=intersect(m,g,fuzz);
    if(t.length > 0) return t[1];
    real ot;
    static real eps=sqrt(realEpsilon);
    real abmax=abs(max(g)-m), abmin=abs(min(g)-m);
    real initr=abs(m-midpoint(g));
    real maxR=2*max(abmax,abmin), step=eps, r=initr;
    real shx=1e-4;
    transform T=shift(m);
    path ig;
    if(t.length > 0) ot=t[1];
    real rm=0, rM=r;
    while(rM-rm > eps) {
      r=(rm+rM)/2;
      t=intersect(T*scale(r)*unitcircle,g,fuzz);
      if(t.length <= 0) {
        rm=r;
      } else {
        rM=r;
        ot=t[1];
      }
    }
    return ot;
  }

  path addnode(path g, real t)
  {// Add a node to 'g' at point(g,t).
    real l=length(g);
    real rt=t % 1;
    if(l == 0 || (t > l && !cyclic(g)) || rt == 0) return g;
    if(cyclic(g)) t=t % l;
    int t0=floor(t);
    int t1=t0+1;
    pair z0=point(g,t0), z1=point(g,t1),
      c0=postcontrol(g,t0), c1=precontrol(g,t1),
      m0=(1-rt)*z0+rt*c0, m1=(1-rt)*c0+rt*c1,
      m2=(1-rt)*c1+rt*z1, m3=(1-rt)*m0+rt*m1,
      m4=(1-rt)*m1+rt*m2;
    guide og=subpath(g,0,t0)..controls m0 and m3..point(g,t);
    if(cyclic(g)) {
      if(t1 < l)
        og=og..controls m4 and m2..subpath(g,t1,l)&cycle;
      else og=og..controls m4 and m2..cycle;
    } else og=og..controls m4 and m2..subpath(g,t1,l);
    return og;
  }

  path addnodes(path g, real fuzz=trembleFuzz()...pair[] P)
  {
    pair[] P=single(P);
    if(length(g) == 0 || P.length == 0 || magneticRadius <= 0) return g;
    path og=g;
    for(pair tp: P) {
      real t=atime(tp,og,fuzz);
      real d=abs(tp-point(og,t));
      if(d < magneticRadius) og=addnode(og,t);
    }
    return og;
  }

  path addnodes(path g, int n)
  {// Add 'n' nodes between each node of 'g'.
    real l=length(g);
    if(n == 0 || l == 0) return g;
    path og=g;
    int np=0;
    for(int i=0; i < l; ++i) {
      real step=1/(n+1);
      for(int j=0; j < n; ++j) {
        og=addnode(og,i*(n+1)+j+step);
        step=1/(n-j);
      }
    }
    return og;
  }

  void operator init(real angle=trembleAngle, real frequency=trembleFrequency,
                     real random=trembleRandom, real fuzz=trembleFuzz()) {
    this.angle=angle;
    this.frequency=frequency;
    this.random=random;
    this.fuzz=fuzz;
  }
  
  path deform(path g...pair[] magneticPoints) {
    /* Return g as it was handwriting.
       The postcontrols and precontrols of the nodes of g will be rotated
       by an angle proportional to 'angle'(in degrees).
       If frequency < 1, floor(1/frequency) nodes will be added to g to
       increase the control points.
       If frequency>= 1, one point for floor(frequency) will be used to deform
       the path.
       'random' controls the randomized coefficient which will be multiplied
       by 'angle'.
       random is 0 means don't use randomized coefficient;
       The higher 'random' is, the more the trembling is randomized. */
    if(length(g) == 0) return g;
    g=addnodes(g,fuzz*abs(max(g)-min(g))...magneticPoints);
    path tg=g;
    frequency=abs(frequency);
    int f=abs(floor(1/frequency)-1);
    tg=addnodes(tg,f);
    int frequency=floor(frequency);
    int tf=(frequency == 0) ? 1 : frequency;
    int l=length(tg);
    guide og=point(tg,0);
    random=abs(random);
    int rsgn(real x){
      int d2=floor(100*x)-10*floor(10*x);
      if(d2 == 0) return 1;
      return 2 % d2 == 0 ? 1 : -1;
    }
    real randf()
    {
      real or;
      if(random != 0) {
        if(1 % tf != 0) or=0;
        else {
          real ur=unitrand();
          or=rsgn(ur)*angle*(1+ur^(1/random));
        }
      } else or=rsgn(unitrand())*1.5*angle;
      return or;
    }

    real first=randf();
    for(int i=1; i <= l; ++i) {
      pair P=point(tg,i);
      real a=randf();
      pair post=rotate(a,point(tg,i-1))*postcontrol(tg,i-1);
      pair pre=rotate((a+randf())/2,P)*precontrol(tg,i);
      if(i == l && (cyclic(tg)))
        og=og..controls post and pre..cycle;
      else
        og=og..controls post and pre..P;
    }
    return og;
  }
}
