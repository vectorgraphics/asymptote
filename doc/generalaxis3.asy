import graph;
import three;
import graph3;
size(0,100);

path3 G=xscale3(1)*(yscale3(2)*unitcircle3);

axis(Label("C",align=Relative(5E)),G,
     LeftTicks(end=false,endlabel=false,8),
     tickspec(0,360,new real(real v) {
		pair d=dir(v);
		real T=intersect(G,O--abs(max(G)-min(G))*(d.x,d.y,0)).x;
		triple v=point(G,T);
		pair z=v;
		pair dir=dir(v,dir(G,T));
		path g=G;
		return intersect(g,z-0.05I*dir--z+0.05I*dir).x;
	      },perpendicular(G,Z)));
