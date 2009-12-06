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

bounds range=image(f,Automatic,a,b,N,Palette);
    
// Major contours

real[] Cvals=uniform(range.min,range.max,Divs);
draw(contour(f,a,b,Cvals,N,operator --),Tickpen);

// Minor contours
real[] cvals;
for(int i=0; i < Cvals.length-1; ++i)
  cvals.append(uniform(Cvals[i],Cvals[i+1],divs)[1:divs]);
draw(contour(f,a,b,cvals,N,operator --),tickpen);

xaxis("$x$",BottomTop,LeftTicks,above=true);
yaxis("$y$",LeftRight,RightTicks,above=true);

palette("$f(x,y)$",range,point(NW)+(0,0.5),point(NE)+(0,1),Top,Palette,
        PaletteTicks(N=Divs,n=divs,Tickpen,tickpen));
