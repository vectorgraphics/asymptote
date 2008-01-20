import graph;

size(300,IgnoreAspect);

real f(real x) {return gamma(x);}
pair F(real x) {return (x,f(x));}

real eps=0.001;
draw(graph(f,eps,4,operator ..),red);
for(int i=1; i < 5; ++i)
  draw(graph(f,-i+eps,-i+1-eps,operator ..),red);

scale(false);
xlimits(-4,4);
ylimits(-6,6);
crop();

xaxis("$x$",RightTicks(NoZero));
yaxis(LeftTicks(NoZero));

label("$\Gamma(x)$",(1,2),red);
