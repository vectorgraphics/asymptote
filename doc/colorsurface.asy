import graph3;
import palette;

size(200,0);

currentprojection=perspective(10,8,4);

real f(pair z) {return 0.5+exp(-abs(z)^2);}

real height(triple v) {return v.z;}

surface s=surface(f,(-1,-1),(1,1),nx=5,Spline);

draw(s,s.palette(height,Gradient(blue,red)),meshpen=black+thick(),nolight);

xaxis3(Label("$x$"),red,Arrow3);
yaxis3(Label("$y$"),red,Arrow3);
zaxis3(XYZero(extend=true),red,Arrow3);
