import graph;
size(0,100);

guide g=ellipse((0,0),1,2);
axis(g,LeftTicks(end=false,endlabel=false,8,"$%.0f$"),
     tickspec(0,360,new real(real v) {
		return intersect(g,(0,0)--abs(max(g)-min(g))*dir(v)).x;}));
