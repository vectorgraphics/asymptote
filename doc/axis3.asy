import graph;
import graph3;
import three;
size(200,0);

currentprojection=perspective(5,2,2);

defaultpen(overwrite(SuppressQuiet));

limits L=autolimits(O,X+Y+Z);

xaxis("$x$",L.O,L.X,red,RightTicks(2,2));
yaxis("$y$",L.O,L.Y,red,RightTicks(2,2));
zaxis("$z$",L.O,L.Z,red,RightTicks(2,2));

