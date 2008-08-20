import grid3;

size(8cm,0,IgnoreAspect);
currentprojection=orthographic(0.5,1,0.5);

scale(Linear, Linear, Log);

draw((-2,-2,0),invisible);
draw((0,2,2),invisible);

grid3(XYZgrid);

xaxis3(Label("$x$",position=EndPoint,align=S),Bounds(Min,Min),RightTicks3());
yaxis3(Label("$y$",position=EndPoint,align=S),Bounds(Min,Min),RightTicks3());
zaxis3(Label("$z$",position=EndPoint,align=(0,0.5)+W),Bounds(Min,Min),
       RightTicks3(beginlabel=false));

