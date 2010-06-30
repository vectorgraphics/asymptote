size(16cm);
import bezulate;

pen edgepen=linewidth(1)+blue;
pen dotpen=deepgreen;
pen labelpen=fontsize(8pt);

path outer = (0.5,5){E}..(5,-1){S}..{W}(4,-4)..{W}(2.5,-1.5){W}..(-0.3,-2.5){W}..(-3,0)..cycle;
outer = subdivide(outer);
path[] p = {outer,shift(-0.5,1.0)*rotate(-22)*scale(1.5,2.4)*subdivide(unitcircle),shift(2.3,0.3)*scale(0.7)*unitcircle};

// a
filldraw(p,lightgrey+evenodd);

real w = 1.1*(max(p).x-min(p).x);

// b
p = shift(w)*p;
draw(p);
path l = point(p[1],2)--point(p[0],4);
draw(l,red);
for(int i = 0; i < p.length; ++i)
{
  real[][] ts = intersections(l,p[i]);
  for(real[] t:ts)
    dot(point(l,t[0]));
}
path l2 = point(l,intersections(l,p[0])[0][0])--point(l,intersections(l,p[2])[1][0]);
real to = intersections(l,p[0])[0][1];
real ti = intersections(l,p[2])[1][1];
draw(l2,edgepen);
label("$A$",point(l2,1),2E,labelpen);
label("$B$",point(l2,0),1.5E,labelpen);

// c
p = shift(w)*p;
l2 = shift(w)*l2;
draw(p);
real timeoffset=2;
path t1=subpath(p[0],to,to+timeoffset);
t1=t1--point(p[2],ti)--cycle;
fill(t1,lightgrey);
draw(point(p[2],ti)--point(p[0],to+4),red);
dot(Label("$A$",labelpen),point(p[2],ti),2E,dotpen);
dot(Label("$B$",labelpen),point(p[0],to),1.5E,dotpen);
dot(Label("$C$",labelpen),point(p[0],to+timeoffset),1.5S,dotpen);
draw(t1,edgepen);
dot(point(p[0],to+4));
draw(shift(-0.5,-0.5)*subpath(p[0],to+4,to+timeoffset+0.5),Arrow(4));

// d
p = shift(w)*p;
p[0] = subpath(p[0],to+timeoffset,to+length(p[0]))--uncycle(p[2],ti)--cycle;
p.delete(2);
draw(p);

// e
p = shift(w)*p;
path q=point(p[1],0)--subpath(p[0],15.4,16)--cycle;
p[0] = subpath(p[0],16,15.4+length(p[0]))--uncycle(p[1],0)--cycle;
p.delete(1);
filldraw(p,lightgrey);

// f
p = shift(w)*p;
filldraw(bezulate(p),lightgrey);
filldraw(shift(3w)*t1,lightgrey);
filldraw(shift(w)*q,lightgrey);


real x = min(p).x - 4.5w;
string l = "abcdef";
for(int i = 0; i < 6; ++i)
{
  label("("+substr(l,i,1)+")",(x,min(p).y),3S,fontsize(10pt));
  x += w;
}
