import graph;
import palette;
import contour;

size(12cm,IgnoreAspect);

pair a=(pi/2,0);
pair b=(3pi/2,2pi);

real f(real x, real y) {return cos(x)*sin(y);}

int N=100;
int Divs=10;

defaultpen(1bp);

bounds range=bounds(-1,1);
    
real[] Cvals=uniform(range.min,range.max,Divs);
guide[][] g=contour(f,a,b,Cvals,N,operator --);

pen[] Palette=quantize(Rainbow(),Divs);

pen[][] interior=interior(g,extend(Palette,grey,black));
fill(g,interior);
draw(g);

palette("$f(x,y)$",range,point(SE)+(0.5,0),point(NE)+(1,0),Right,Palette,
        PaletteTicks("$%+#0.1f$",N=Divs));
