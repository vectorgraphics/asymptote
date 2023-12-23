import graph;

size(10cm,0);

real xmin=-4,xmax=4;
real ymin=-2,ymax=10;

real f(real x) {return x^2;}

marker mark=marker(scale(4)*plus,
                    markuniform(new pair(real t) {return Scale((t,f(t)));},
                                xmin,xmax,round(2*(xmax-xmin))),1bp+red);

draw(graph(f,xmin,xmax,n=400),linewidth(1bp),mark);

ylimits(-2.5,10,Crop);

xaxis(Label("$x$",position=EndPoint, align=NE),xmin=xmin,xmax=xmax,
      Ticks(scale(.7)*Label(align=E),NoZero,begin=false,beginlabel=false,
            end=false,endlabel=false,Step=1,step=.25,
            Size=1mm, size=.5mm,pTick=black,ptick=gray),Arrow);

yaxis(Label("$y$",position=EndPoint, align=NE),ymin=ymin,ymax=ymax,
      Ticks(scale(.7)*Label(),NoZero,begin=false,beginlabel=false,
            end=false,endlabel=false,Step=1,step=.25,Size=1mm,size=.5mm,
            pTick=black,ptick=gray),Arrow);
