import three;
size(200,0);

currentprojection=perspective((4,4,3));

triple f(path p, real position) {
  pair z=point(p,position);
  return (z.x,z.y,position/length(p));
}

real r=1.5;
draw("$x$",(0,0,0)--(r,0,0),1,red,Arrow);
draw("$y$",(0,0,0)--(0,r,0),1,red,Arrow);
draw("$z$",(0,0,0)--(0,0,r),1,red,Arrow);
  
draw(graph(f,E..N..W..S..E..N..W..S),blue);
