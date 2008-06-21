import graph;
import palette;
import contour;

size(200);

int n=100;

real[] x=new real[n];
real[] y=new real[n];
real[] f=new real[n];

real F(real a, real b) {return a^2+b^2;}

real r() {return 1.1*(rand()/randMax*2-1);}

for(int i=0; i < n; ++i) {
  x[i]=r();
  y[i]=r();
  f[i]=F(x[i],y[i]);
}

pen Tickpen=black;
pen tickpen=gray+0.5*linewidth(currentpen);
pen[] Palette=BWRainbow();

bounds range=image(x,y,f,Range(0,2),Palette);
draw(contour(pairs(x,y),f,new real[]{0.25,0.5,1},operator ..));

palette("$f(x,y)$",range,point(NW)+(0,0.5),point(NE)+(0,0.8),Top,Palette,
        PaletteTicks(Tickpen,tickpen));
