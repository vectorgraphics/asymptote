import graph;
import palette;
import contour;

size(10cm,10cm,IgnoreAspect);

pair a=(0,0);
pair b=(2pi,2pi);

real f(real x, real y) {return cos(x)*sin(y);}

int N=200;
int Divs=10;
int divs=2;

defaultpen(1bp);
pen Tickpen=black;
pen tickpen=gray+0.5*linewidth(currentpen);
pen[] Palette=BWRainbow();

scale(false);

bounds range=image(f,Automatic,a,b,N,Palette);
    
xaxis("$x$",BottomTop,LeftTicks,Above);
yaxis("$y$",LeftRight,RightTicks,Above);

// Major contours
real[] Cvals;
Cvals=sequence(11)/10 * (range.max-range.min) + range.min;
draw(contour(f,a,b,Cvals,N,operator ..),Tickpen);

// Minor contours
real[] cvals;
real[] sumarr=sequence(1,divs-1)/divs * (range.max-range.min)/Divs;
for (int ival=0; ival < Cvals.length-1; ++ival)
    cvals.append(Cvals[ival]+sumarr);
draw(contour(f,a,b,cvals,N,operator ..),tickpen);

palette("$f(x,y)$",range,point(NW)+(0,0.5),point(NE)+(0,1),Top,Palette,
	PaletteTicks(N=Divs,n=divs,Tickpen,tickpen));


