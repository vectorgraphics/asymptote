import graph3;

currentlight=adobe;

size(12cm,0);

real f(pair z) {return min(sqrt(1-z.x^2),sqrt(1-z.y^2));}

surface s=surface(f,(-1,-1),(1,1),100,Spline);
draw(s,blue);
draw(zscale3(-1)*s,blue);
