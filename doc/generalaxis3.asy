import graph3;

size(0,100);

path3 g=yscale3(2)*unitcircle3;
currentprojection=perspective(10,10,10);

axis(Label("C",position=0,align=15X),g,InTicks(endlabel=false,8,end=false),
     ticklocate(0,360,new real(real v) {
         path3 h=O--max(abs(max(g)),abs(min(g)))*dir(90,v);
         return intersect(g,h)[0];},
       new triple(real t) {return cross(dir(g,t),Z);}));

