import graph;
import palette;

int n=256;
real ninv=2pi/n;
real[][] v=new real[n][n];

for(int i=0; i < n; ++i)
  for(int j=0; j < n; ++j)
    v[i][j]=sin(i*ninv)*cos(j*ninv);

pen[] Palette=BWRainbow();

picture plot;

image(plot,v,Palette,(0,0),(1,1));
picture bar=palette(v,5mm,Palette,"$A$",PaletteTicks("$%+#.1f$"));

add(plot.fit(250,250),W);
add((1cm,0),bar.fit(0,250),E);

