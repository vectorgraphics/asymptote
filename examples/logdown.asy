import graph;
size(200,IgnoreAspect);

real log10Down(real x) {return -log10(x);}
real pow10Down(real x) {return pow10(-x);}

scaleT LogDown=scaleT(log10Down,pow10Down,logarithmic=true);
scale(Linear,LogDown);

draw(graph(exp,-5,5));

yaxis("$y$",RightTicks(Label(Fill(white)),DefaultLogFormat),BeginArrow);
xaxis("$x$",LeftTicks(NoZero),EndArrow);
