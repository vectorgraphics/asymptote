import three;

size(100,0);
guide3 g=(1,0,0)..(0,1,1)..(-1,0,0)..(0,-1,1)..cycle;
filldraw(g,lightgrey);
draw(((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle));
dot(g,red);
