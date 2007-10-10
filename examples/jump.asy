import graph;
size(4inches,0);

real f1(real x) {return (1+x^2);} 
real f2(real x) {return (4-x);}

xaxis("$x$",LeftTicks,Arrow);
yaxis("$y$",RightTicks,Arrow);

draw("$y=1+x^2$",graph(f1,-2,1)); 
dot((1,f1(1)),UnFill);

draw("$y=4-x$",graph(f2,1,5),LeftSide,red,Arrow);
dot((1,f2(1)),red);
