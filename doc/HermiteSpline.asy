import graph;

size(140mm,70mm,IgnoreAspect); 
scale(false); 
real[] x={1,3,4,5,6};
real[] y={1,5,2,0,4}; 

marker mark=marker(scale(1mm)*cross(6,false,r=0.35),red,Fill); 

draw(graph(x,y,Hermite),"Hermite Spline",mark);
xaxis("$x$",Bottom,LeftTicks(x)); 
yaxis("$y$",Left,LeftTicks); 
attach(legend(),point(NW),40S+30E,UnFill);

