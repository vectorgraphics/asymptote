import tube;
import graph3;
import palette;

size(10cm,0);
currentprojection=perspective(1,1,1,up=-Y);

int e=1;
real x(real t) {return cos(t)+2*cos(2t);}
real y(real t) {return sin(t)-2*sin(2t);}
real z(real t) {return 2*e*sin(3t);}

path3 p=scale3(2)*graph(x,y,z,0,2pi,50,operator ..)&cycle;

pen[] pens=Gradient(6,red,blue,purple);
pens.push(yellow);
for (int i=pens.length-2; i >= 0 ; --i)
  pens.push(pens[i]);

path sec=subpath(Circle(0,1.5,2*pens.length),0,pens.length);
coloredpath colorsec=coloredpath(sec, pens,colortype=coloredNodes);
draw(tube(p,colorsec));
