import graph;
import three;
import graph3;
size(0,100);

path3 G=xscale3(1)*(yscale3(2)*unitcircle3);

projection P=currentprojection;

axis(G,LeftTicks(end=false,endlabel=false,8,"$%.0f$"),
     tickspec(0,360,new real(real v) {
		pair d=dir(v);
		real T=intersect(G,O--abs(max(G)-min(G))*(d.x,d.y,0)).x;
		pair z=project(point(G,T),P);
		pair dir=project(dir(G,T),P);
		path g=G;
		return intersect(g,z-0.5I*dir--z+0.5I*dir).x;
	      },perpendicular(G,Z)));

