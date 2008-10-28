size(6cm,0);
import bsp;

real u=2.5;
real v=1;

currentprojection=oblique;

path3 y=plane((2u,0,0),(0,2v,0),(-u,-v,0));
path3 l=rotate(90,Z)*rotate(90,Y)*y;
path3 g=rotate(90,X)*rotate(90,Y)*y;

face[] faces;
pen[] p={red,green,blue,black};
int[] edges={0,0,0,2};
gouraudshade(faces.push(y),project(y),p,edges);
gouraudshade(faces.push(l),project(l),p,edges);
gouraudshade(faces.push(g),project(g),new pen[]{cyan,magenta,yellow,black},
	     edges);

add(faces);

