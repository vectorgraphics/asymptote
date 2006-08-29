import graph3;
import solids;
size(0,150);
currentprojection=perspective(0,0,10);
currentlight=(2,0,0);
pen color1=green;
pen color2=red;
real alpha=250;

real f(real x) {return 2x^2-x^3;}
triple F(real x) {return (x,f(x),0);}

ngraph=12;
guide3[] p=new guide3[] {
  graph(F,0.7476,1.8043)--cycle3,
  graph(F,0.7,0.7476)--graph(F,1.7787,1.8043)--cycle3,
  graph(F,0,0.7)--graph(F,1.8043,2)--cycle3};

pen[] pn=new pen[] {color1,color2,color1};

for(int i=0; i < p.length; ++i) {
  revolution a=revolution(p[i],Y,0,alpha);
  a.fill(10,pn[i]);
  filldraw(p[i],pn[i]);
  filldraw(rotate(alpha,Y)*p[i],pn[i]);
}
bbox3 b=autolimits(O,2.1*(X+Y/1.5)+Z);

draw((4/3,0,0)--F(4/3),dashed);
xtick("$\frac{4}{3}$",(4/3,0,0));

xaxis(Label("$x$",1),b,Arrow);
yaxis(Label("$y$",1),b,dashed,Arrow);
arrow("$y=2x^2-x^3$",F(1.6),NE,0.4cm);
draw(arc(1.18Y,0.3,90,0,7.5,180),ArcArrow);
