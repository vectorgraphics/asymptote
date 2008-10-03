import graph;
size(0,100);

path g=ellipse((0,0),1,2);

scale(true);

axis(Label("C",align=10W),g,LeftTicks(endlabel=false,8,end=false),
     ticklocate(0,360,new real(real v) {
         path h=(0,0)--max(abs(max(g)),abs(min(g)))*dir(v);
         return intersect(g,h)[0];}));
