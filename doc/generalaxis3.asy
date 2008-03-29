import graph3;

size(0,100);

path3 G=xscale3(1)*(yscale3(2)*unitcircle3);

axis(Label("C",align=Relative(5E)),G,
     LeftTicks(endlabel=false,8,end=false),
     ticklocate(0,360,new real(real v) {
         path g=G;
         path h=O--max(abs(max(G)),abs(min(G)))*dir(90,v);
         return intersect(g,h)[0];
       },new pair(real t) {
	 t /= ninterpolate;
	 return dir(point(G,t),cross(dir(G,t),Z));}));
