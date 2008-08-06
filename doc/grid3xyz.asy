import grid3;

size(8cm,0);
currentprojection=orthographic(0.5,1,0.5);

defaultpen(overwrite(SuppressQuiet));
bbox3 b=limits((-2,-2,0),(0,2,2));

scale(Linear, Linear, Log);

grid3(b,XYZgrid);
xaxis(Label("$x$",position=EndPoint,align=S),b,RightTicks());
yaxis(Label("$y$",position=EndPoint,align=S),b,RightTicks());
zaxis(Label("$z$",position=EndPoint,align=(0,0.5)+W),b,RightTicks());

