import graph;

size(400,150,IgnoreAspect);

real[] x=sequence(12);
real[] y=sin(2pi*x/12);

scale(false);

string[] month={"Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"};

draw(graph(x,y),red,MarkFill[0]);

xaxis(BottomTop,LeftTicks(new string(real x) {
      return month[round(x % 12)];}));
yaxis("$y$",LeftRight,RightTicks(4));
