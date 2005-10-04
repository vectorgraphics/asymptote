size(200,200,IgnoreAspect);
import graph;

real L=1;
real epsilon=0.25;

real a(int n) {return L+1/n;}

for(int i=1; i < 20; ++i)
  dot((i,a(i)));

real N=1/epsilon;

xaxis(Label("$n$",align=2S));
yaxis(Label("$a_n$",0.85));

xtick("$2$",2);
ytick("$\frac{3}{2}$",3/2);
ytick("$2$",2);

yequals(Label("$L$",0,up),L,extend=true,blue);
yequals(Label("$L+\epsilon$",1,NW),L+epsilon,extend=true,red+dashed);
yequals(Label("$L-\epsilon$",1,SW),L-epsilon,extend=true,red+dashed);

xequals(N,extend=true,darkgreen+dashed);
labelx(shift(0,-10)*"$N=\frac{1}{\epsilon}$",N,E,darkgreen);

label("$a_n=1+\frac{1}{n},\quad \epsilon=\frac{1}{4}$",point((0,1)),10S+E);
