import math;

int n=7;

size(200,0);

draw(unitcircle,red);
for (int i=0; i < n-1; ++i)
  for (int j=i+1; j < n; ++j)
    drawline(zeta(n,i),zeta(n,j),blue);
