import graph;
import three;
import graph3;
size(0,100);

path3 G=xscale3(1)*(yscale3(2)*unitcircle3);

axis(Label("C",align=Relative(5E)),G,
     LeftTicks(end=false,endlabel=false,8),
     tickspec(0,360,new real(real v) {
		path g=G;
		path h=O--max(abs(max(G)),abs(min(G)))*dir(90,v);
		return intersect(g,h).x;
	      },perpendicular(G,Z)));
