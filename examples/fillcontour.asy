import graph;
import palette;
import contour;

size(10cm,10cm,IgnoreAspect);

pair a=(pi/2,0);
pair b=(1.5*pi+epsilon,2pi);

real f(real x, real y) {return cos(x)*sin(y);}

int N=300;
int Divs=10;

defaultpen(1bp);

bounds range=bounds(-1,1);
    
real[] Cvals=sequence(Divs+1)/Divs*(range.max-range.min)+range.min;
guide[][] g=contour(f,a,b,Cvals,N,operator --);

pen[] Palette=quantize(Rainbow(),Divs);

fill(g,Palette);
draw(g);

palette("$f(x,y)$",range,point(NW)+(0,0.5),point(NE)+(0,1),Top,Palette,
	PaletteTicks(N=Divs));
