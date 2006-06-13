import contour;

size(200);
int np=100;
real f(real a, real b){return a^2+b^2;}
pair[] points;
real[] values;

real r() {return 1.2*(rand()/randMax*2-1);}

for(int i=0; i < np; ++i)
  points.push((r(),r()));

for(int i=0; i < np; ++i)
  values.push(f(points[i].x,points[i].y));

draw(contour(points,values,new real[]{0.25,0.5,1},operator..),blue);
