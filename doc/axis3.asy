import three;
import graph;
import graph3;

size(0,200,IgnoreAspect);

currentprojection=perspective(5,2,2);

defaultpen(overwrite(SuppressQuiet));

scale(Linear,Linear,Log(automax=false));

bbox3 b=autolimits(Z,X+Y+30Z);

xaxis("$x$",b,red,RightTicks(2,2));
yaxis("$y$",b,red,RightTicks(2,2));
zaxis("$z$",b,red,RightTicks);

