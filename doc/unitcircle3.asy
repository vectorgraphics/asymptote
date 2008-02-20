import three;
size(100,0);
path3 g=(1,0,0)..(0,1,0)..(-1,0,0)..(0,-1,0)..cycle;
filldraw(g,lightgrey);
draw(O--Z,red+dashed,BeginBar,Arrow);
draw(((-1,-1,0)--(1,-1,0)--(1,1,0)--(-1,1,0)--cycle));
dot(g,red);
