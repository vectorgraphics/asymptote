import graph;
size(8cm,6cm,IgnoreAspect);

pair S0=(4,0.2);
pair S1=(2,3);
pair S8=(0.5,0);

xaxis("$S$");
yaxis(Label("$I$",0.5));

draw(S0{curl 0}..tension 1.5..S1{W}..tension 1.5..{curl 0}S8,Arrow(Fill,0.4));
draw((S1.x,0)..S1,dashed);
draw((0,S1.y)..S1,dotted);

labelx("$\frac{\gamma}{\beta}$",S1.x);
labelx("$S_\infty$",S8.x);
labely("$I_{\max}$",S1.y);

