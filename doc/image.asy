size(250,250);
import graph;
import palette;

int n=256;
real ninv=2pi/n;
real[][] v=new real[n][n];

for(int i=0; i < n; ++i)
  for(int j=0; j < n; ++j)
    v[i][j]=sin(i*ninv)*cos(j*ninv);

pen[] Palette=BWRainbow();

image(v,Palette,(0,0),(1,1));
addabout((1,0),palette(v,Palette,"$A$",
		       LeftTicks(0.0,0.0,Ticksize,0.0,"%+#.1f")));
