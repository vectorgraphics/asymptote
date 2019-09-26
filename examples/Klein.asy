import graph3;

size(469pt);

currentprojection=perspective(
camera=(25.0851928432063,-30.3337528952473,19.3728775115443),
up=Z,
target=(-0.590622314050054,0.692357205025578,-0.627122488455679),
zoom=1,
autoadjust=false);

triple f(pair t) {
  real u=t.x;
  real v=t.y;
  real r=2-cos(u);
  real x=3*cos(u)*(1+sin(u))+r*cos(v)*(u < pi ? cos(u) : -1);
  real y=8*sin(u)+(u < pi ? r*sin(u)*cos(v) : 0);
  real z=r*sin(v);
  return (x,y,z);
}

surface s=surface(f,(0,0),(2pi,2pi),8,8,Spline);
draw(s,lightolive+white,"bottle",render(merge=true));

string lo="$\displaystyle u\in[0,\pi]: \cases{x=3\cos u(1+\sin u)+(2-\cos u)\cos u\cos v,\cr
y=8\sin u+(2-\cos u)\sin u\cos v,\cr
z=(2-\cos u)\sin v.\cr}$";

string hi="$\displaystyle u\in[\pi,2\pi]:\\\cases{x=3\cos u(1+\sin u)-(2-\cos u)\cos v,\cr
y=8\sin u,\cr
z=(2-\cos u)\sin v.\cr}$";

real h=0.0125;

begingroup3("parametrization");
draw(surface(xscale(-0.38)*yscale(-0.18)*lo,s,0,1.7,h,bottom=false),
     "[0,pi]");
draw(surface(xscale(0.26)*yscale(0.1)*rotate(90)*hi,s,4.9,1.4,h,bottom=false),
     "[pi,2pi]");
endgroup3();

begingroup3("boundary");
draw(s.uequals(0),blue+dashed);
draw(s.uequals(pi),blue+dashed);
endgroup3();

add(new void(frame f, transform3 t, picture pic, projection P) {
    draw(f,invert(box(min(f,P),max(f,P)),P),"frame");
  });
