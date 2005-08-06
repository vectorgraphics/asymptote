import graph;
size(0,100);

guide g=ellipse((0,0),100,200);
axis(g,0,0.5*arclength(g),RightTicks(8,"$%.0f$"),degrees);
