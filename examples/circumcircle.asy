unitsize(1inch);

path tri=(0,0)--(1,0)--(2,1)--cycle;
pair p1=point(tri,0.5);
pair p2=point(tri,1.5);
pair z0=extension(p1,p1+I*dir(tri,0.5),p2,p2+I*dir(tri,1.5));
dot(z0);
draw(circle(z0,abs(z0-point(tri,0))));
draw(tri,red);
