import graph;
size(0,100);

guide g=ellipse((0,0),1,2);
axis(Label("C",align=10W),g,LeftTicks(end=false,endlabel=false,8),
     ticklocate(0,360,new real(real v) {
		  path h=(0,0)--max(abs(max(g)),abs(min(g)))*dir(v);
		  return intersect(g,h).x;}));
