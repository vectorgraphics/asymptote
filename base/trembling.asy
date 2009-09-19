// Copyright (c) 2008, Philippe Ivaldi.
// http://www.piprime.fr/
// trembling.asy: handwriting package for the software Asymptote.

// This program is free software ; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation ; either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY ; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with this program ; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

// COMMENTARY:

// THANKS:

// BUGS:
// magnetic points are experimental...

// CODE:

import geometry;
import stats;

/*<asyxml><variable type="pair[]" signature="magneticPoints"><code></asyxml>*/
pair[] magneticPoints;
/*<asyxml></code><documentation>Array of magnetic points.
  When a point is magnetized, a trembled path will pass though P if the original path passes through P.
  When trembling is enabled with the routine 'startTrembling' all the drawn points is added to this array.
  For convenience, one can magnetize an arbitrary number of points with the routine 'magnetize'.<look href="#magnetize(...point[])"><look href="#startTrembling(real,real,real,real,bool">
  </documentation></variable></asyxml>*/
real magneticRadius=1; // unit is bp in postscript coordinates.
real trembleFuzz(){return min(1e-3,magneticRadius/10);}
// real trembleFuzz(){return min(1e-1,1);}

pair[] single(pair[] P)
{
  pair[] op;
  bool allow;
  for (int i=0; i < P.length-1; ++i) {
    allow=true;
    for (int j=i+1; j < P.length; ++j) {
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

/*<asyxml><function type="pair" signature="attract(pair,path,real)"><code></asyxml>*/
real atime(pair m, path g, real fuzz=trembleFuzz())
{/*<asyxml></code><documentation>Return the time of the nearest point of 'm' which is on the path g.
   'fuzz' is the argument 'fuzz' of 'intersect'.</documentation></function></asyxml>*/
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
  // do {
  //   do {// Find a radius for intersection
  //     r += step;
  //     ig=T*scale(r)*unitcircle;
  //     if(ig == g) {
  //       T *= shift(shx,0);
  //       ig=T*scale(r)*unitcircle;
  //     }
  //     t=intersect(ig,g);
  //   } while(t.length <= 0 && r <= maxR && inside(g,ig) != 1);//
  //   if(t.length <= 0) { // degenerated case
  //     r=initr;
  //     T *= shift(shx,0);
  //     warning("atime","atime needs numerical adjustment.",position=true);
  //   }
  // } while(t.length <= 0);
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

/*<asyxml><function type="void" signature="magnetize(...pair[])"><code></asyxml>*/
void magnetize(...pair[] P)
{/*<asyxml></code><documentation>Magnetize the points P.
   When a point is magnetized, a trembled path will pass though P if the original path passes through P.</documentation></function></asyxml>*/
  if(magneticRadius <= 0) return;
  currentpicture.uptodate=false;
  currentpicture.nodes.insert(0,
                              new void(frame f, transform t, transform T, pair lb, pair rt)
                              {
                                for (pair PT:P) {
                                  magneticPoints.push(t*PT);
                                }
                              });
}

/*<asyxml><function type="void" signature="magnetize(...triangle[])"><code></asyxml>*/
void magnetize(...triangle[] t)
{/*<asyxml></code><documentation>Magnetize the vertices of the triangles t.<look href="#magnetize(...pair[])"/></documentation></function></asyxml>*/
  for(int i=0; i < t.length; ++i)
    magnetize(...new point[] {t[i].A, t[i].B, t[i].C});
}

void trueMagnetize(pair p)
{
  if(magneticRadius <= 0) return;
  magneticPoints.push(p);
}

/*<asyxml><function type="path" signature="addnode(path,real)"><code></asyxml>*/
path addnode(path g, real t)
{/*<asyxml></code><documentation>Add a node to 'g' at point(g,t).</documentation></function></asyxml>*/
  real l=length(g);
  real rt=t%1;
  if (l==0 || (t > l && !cyclic(g)) || rt == 0) return g;
  if(cyclic(g)) t=t%l;
  int t0=floor(t);
  int t1=t0+1;
  pair z0=point(g,t0),    z1=point(g,t1),
    c0=postcontrol(g,t0), c1=precontrol(g,t1),
    m0=(1-rt)*z0+rt*c0,   m1=(1-rt)*c0+rt*c1,
    m2=(1-rt)*c1+rt*z1,   m3=(1-rt)*m0+rt*m1,
    m4=(1-rt)*m1+rt*m2;
  guide og=subpath(g,0,t0);
  og=og..controls m0 and m3..point(g,t);
  if(cyclic(g))
    if(t1 < l) {
      og=og..controls m4 and m2..subpath(g,t1,l)&cycle;
    } else og=og..controls m4 and m2..cycle;
  else og=og..controls m4 and m2..subpath(g,t1,l);
  return og;
}

path addnodes(path g, real fuzz=trembleFuzz() ...pair[] P)
{
  pair[] P=single(P);
  if(length(g) == 0 || P.length == 0 || magneticRadius <= 0) return g;
  path og=g;
  for(pair tp:P) {
    real t=atime(tp,og,fuzz);
    real d=abs(tp-point(og,t));
    if(d < magneticRadius) og=addnode(og,t);
  }
  return og;
}

/*<asyxml><function type="path" signature="addnodes(path,int)"><code></asyxml>*/
path addnodes(path g, int n)
{/*<asyxml></code><documentation>Add 'n' nodes between each node of 'g'.</documentation></function></asyxml>*/
  real l=length(g);
  if(n == 0 || l == 0) return g;
  path og=g;
  int np=0;
  for (int i=0; i < l; ++i) {
    real step=1/(n+1);
    for (int j=0; j < n; ++j) {
      og=addnode(og,i*(n+1)+j+step);
      step=1/(n-j);
    }
  }
  return og;
}



/*<asyxml><variable type="real" signature="trembleAngle"><code></asyxml>*/
real trembleAngle=4, trembleFrequency=0.5, trembleRandom=2;/*<asyxml></code><documentation>Variables used by the routine 'tremble'.</documentation></variable></asyxml>*/

/*<asyxml><function type="path" signature="tremble(path,real,real,real,real)"><code></asyxml>*/
path tremble(path g,
             real angle=trembleAngle,
             real frequency=trembleFrequency,
             real random=trembleRandom,
             real fuzz=trembleFuzz())
{/*<asyxml></code><documentation>Return g as it was handwriting.
   The postcontrols and precontrols of the nodes of g will be rotated
   by an angle proportional to 'angle' (in degrees).
   If frequency < 1, floor(1/frequency) nodes will be added to g to increase the
   control points.
   If frequency >= 1, one point for floor(frequency) will be used to deform the path.
   'random' controls the randomized coefficient which will be multiplied 'angle'.
   random is 0 means don't use randomized coefficient;
   More 'random' is hight more the coefficient is hight and the trembling seems randomized.</documentation></function></asyxml>*/
  if(length(g) == 0) return g;
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
    return 2%d2 == 0 ? 1 : -1;
  }
  real randf()
  {
    real or;
    if(random != 0)
      if(1%tf != 0) or=0; else {
        real ur=unitrand();
        or=rsgn(ur)*angle*(1+ur^(1/random));
      }
    else or=rsgn(unitrand())*1.5*angle;
    return or;
  }

  real first=randf(), a=first;
  real gle;
  for(int i=1; i <= l; ++i)
    {
      pair P=point(tg,i);
      // gle=(i == l && (cyclic(tg))) ? first : a;
      // pair post=rotate(gle,point(tg,i-1))*postcontrol(tg,i-1);
      // pair pre=rotate(gle,P)*precontrol(tg,i);
      a=randf();
      pair post=rotate(a,point(tg,i-1))*postcontrol(tg,i-1);
      pair pre=rotate((a+randf())/2,P)*precontrol(tg,i);
      if(i == l && (cyclic(tg)))
        og=og..controls post and pre..cycle;
      else
        og=og..controls post and pre..P;
    }
  return og;
}

typedef path pathModifier(path);
pathModifier NoModifier=new path(path g){return g;};
pathModifier tremble(real angle=trembleAngle,
                     real frequency=trembleFrequency,
                     real random=trembleRandom,
                     real fuzz=trembleFuzz())
{
  return new path(path g){return tremble(g,angle,frequency,random,fuzz);};
}

void orig_draw(frame f, path g, pen p=currentpen)=draw;
void tremble_draw(frame f, path g, pen p=currentpen){}

void tremble_marknodes(picture pic=currentpicture, frame f, path g) {}
void tremble_markuniform(bool centered=false, int n, bool rotated=false) {}

int tremble_circlenodesnumber(real r){return 5;}
int orig_circlenodesnumber(real r)=circlenodesnumber;

int tremble_circlenodesnumber1(real r, real angle1, real angle2){return 4;}
int orig_circlenodesnumber1(real r, real angle1, real angle2)=circlenodesnumber;

int tremble_ellipsenodesnumber(real a, real b){return 50;}
int orig_ellipsenodesnumber(real a, real b)=ellipsenodesnumber;

int tremble_ellipsenodesnumber1(real a, real b, real angle1, real angle2, bool dir){return 20;}
int orig_ellipsenodesnumber1(real a, real b, real angle1, real angle2, bool dir)=ellipsenodesnumber;

int tremble_parabolanodesnumber(parabola p, real angle1, real angle2){return 20;}
int orig_parabolanodesnumber(parabola p, real angle1, real angle2)=parabolanodesnumber;

int tremble_hyperbolanodesnumber(hyperbola h, real angle1, real angle2){return 20;}
int orig_hyperbolanodesnumber(hyperbola h, real angle1, real angle2)=hyperbolanodesnumber;

restricted bool tremblingMode=false;
/*<asyxml><function type="void" signature="startTrembling(real,real,real,real,bool"><code></asyxml>*/
void startTrembling(real angle=trembleAngle,
                    real frequency=trembleFrequency,
                    real random=trembleRandom,
                    real fuzz=trembleFuzz(),
                    bool magnetizePoints=true)
{/*<asyxml></code><documentation>Calling this routine all drawn paths will be trembled with the givens parameters.<look href="#tremble(path,real,real,real,real)"/>
   If 'magnetizePoints' is true, most dotted points are automatically magnetized.<look href="magnetize(...pair[])"/></documentation></function></asyxml>*/
  if(!tremblingMode) {
    tremblingMode=true;
    tremble_draw=new void(frame f, path g, pen p=currentpen)
      {
        if(length(g) == 0 && magnetizePoints) {
          trueMagnetize(point(g,0));
        }
        g=addnodes(g,fuzz*abs(max(g)-min(g)) ...magneticPoints);
        orig_draw(f,tremble(g,angle,frequency,random,fuzz),p);
      };

    plain.draw=tremble_draw;

    if(magnetizePoints) {
      marknodes=new void(picture pic=currentpicture, frame f, path g)
        {
          for(int i=0; i <= length(g); ++i) {
            add(pic,f,point(g,i));
            magnetize(point(g,i));
          }
        };
      dot=dot();

      markuniform=new markroutine(bool centered=false, int n, bool rotated=false)
        {
          return new void(picture pic=currentpicture, frame f, path g) {
            if(n <= 0) return;
            void add(real x) {
              real t=reltime(g,x);
              add(pic,rotated ? rotate(degrees(dir(g,t)))*f : f,point(g,t));
              magnetize(point(g,t));
            }
            if(centered) {
              real width=1/n;
              for(int i=0; i < n; ++i) add((i+0.5)*width);
            } else {
              if(n == 1) add(0.5);
              else {
                real width=1/(n-1);
                for(int i=0; i < n; ++i)
                  add(i*width);
              }
            }
          };
        };
    }

    circlenodesnumber=tremble_circlenodesnumber;
    circlenodesnumber=tremble_circlenodesnumber1;

    ellipsenodesnumber=tremble_ellipsenodesnumber;
    ellipsenodesnumber=tremble_ellipsenodesnumber1;

    parabolanodesnumber=tremble_parabolanodesnumber;
    hyperbolanodesnumber=tremble_hyperbolanodesnumber;
  }
}

// void stopTrembling()
// {
//   if(tremblingMode) {
//     tremblingMode=false;
//     plain.draw=orig_draw;
//     // draw=orig_draw1;
//     circlenodesnumber=orig_circlenodesnumber;
//     circlenodesnumber=orig_circlenodesnumber1;
//     ellipsenodesnumber=orig_ellipsenodesnumber;
//     ellipsenodesnumber=orig_ellipsenodesnumber1;
//     parabolanodesnumber=orig_parabolanodesnumber;
//     hyperbolanodesnumber=orig_hyperbolanodesnumber;
//   }
// }
