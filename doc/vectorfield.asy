import graph;
size(100);

pair a=(0,0);
pair b=(2pi,2pi);

path vector(pair z) {return (sin(z.x),cos(z.y));}

add(vectorfield(vector,a,b));
