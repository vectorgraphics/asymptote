import three;
size(200,0);

currentprojection=perspective((5,4,2));

real f(pair z) {return 0.5+exp(-abs(z)^2);}

draw((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle3);

real r=1.75;
draw("$x$",(0,0,0)--(r,0,0),1,red,Arrow);
draw("$y$",(0,0,0)--(0,r,0),1,red,Arrow);
draw("$z$",(0,0,0)--(0,0,r),1,red,Arrow);
  
draw((1,0,0){Y}..(0,1,0){-X}..(-1,0,0){-Y}..(0,-1,0){X}..cycle3,green);
  
label("$O$",(0,0,0),S,red);
  
add(surface(f,(-1,-1),(1,1),n=10));
