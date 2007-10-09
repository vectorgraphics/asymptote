import graph; 
size(8cm,6cm,IgnoreAspect); 
 
typedef real realfcn(real); 
realfcn F(real p) { 
  return new real(real x) {return sin(p*x);}; 
}; 
 
for(int i=1; i < 5; ++i)
  draw(graph(F(i*pi),0,1),Pen(i),
       "$\sin("+(i == 1 ? "" : (string) i)+"\pi x)$"); 
xaxis("$x$",BottomTop,LeftTicks); 
yaxis("$y$",LeftRight,RightTicks(trailingzero)); 
 
attach(legend(2),(point(S).x,truepoint(S).y),10S,UnFill); 
