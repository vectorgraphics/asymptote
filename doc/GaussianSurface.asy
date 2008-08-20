import graph3;

size(200,0);

currentprojection=perspective(5,4,2);

real f(pair z) {return 0.5+exp(-abs(z)^2);}

draw((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle);

//draw(arc(0.12Z,0.2,90,60,90,15),ArcArrow);
draw(arc(0.12Z,0.2,90,60,90,15));

surface s=surface(f,(-1,-1),(1,1),nx=10);
  
xaxis3(Label("$x$",1),red,Arrow);
yaxis3(Label("$y$",1),red,Arrow);
zaxis3(Label("$z$",1),red,Arrow);

label("$O$",(0,0,0),S,red);

draw(s,nolight);
