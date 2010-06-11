import graph3;

currentprojection=orthographic(5,4,2);

size(12cm,0);

real f(pair z) {return min(sqrt(1-z.x^2),sqrt(1-z.y^2));}

surface s=surface(f,(0,0),(1,1),40,Spline);

transform3 t=rotate(90,O,Z), t2=t*t, t3=t2*t, i=xscale3(-1)*zscale3(-1);
draw(surface(s,t*s,t2*s,t3*s,i*s,i*t*s,i*t2*s,i*t3*s),blue,
     render(compression=Low,closed=true,merge=true));
