size(0,100);
import geometry;

triangle t=triangle(b=3,alpha=90,c=4);
  
dot((0,0));

draw(t);
draw(rotate(90)*t,red);
draw(shift((-4,0))*t,blue);
draw(reflect((0,0),(1,0))*t,green);
draw(slant(2)*t,magenta);
