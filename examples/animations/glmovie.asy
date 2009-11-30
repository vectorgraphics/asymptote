settings.autoplay=true;
settings.loop=true;

import graph3;
import animate;
currentprojection=orthographic(1,-2,0.5);

animation A;
int n=25;

for(int i=0; i < n; ++i) {
  picture pic;
  size3(pic,6cm);
  real k=i/n*pi;
  real f(pair z) {return 4cos(abs(z)-k)*exp(-abs(z)/6);}
  draw(pic,surface(f,(-4pi,-4pi),(4pi,4pi),Spline),paleblue);
  draw(pic,shift(i*6Z/n)*unitsphere,yellow);
  A.add(pic);
}

A.glmovie();
