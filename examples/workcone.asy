import solids;
size(0,150);
currentprojection=orthographic(0,-30,5);

real r=4;
real h=10;
real s=8;
real x=r*s/h;

real sr=5;
real xr=r*sr/h;

real s1=sr-0.1;
real x1=r*s1/h;

real s2=sr+0.2;
real x2=r*s2/h;

render render=render(compression=0,merge=true);

draw(scale(x1,x1,-s1)*shift(-Z)*unitcone,lightblue+opacity(0.5),render);

path3 p=(x2,0,s2)--(x,0,s+0.005);
revolution a=revolution(p,Z);
draw(surface(a),lightblue+opacity(0.5),render);

path3 q=(x,0,s)--(r,0,h);
revolution b=revolution(q,Z);
draw(surface(b),white+opacity(0.5),render);

draw((-r-1,0,0)--(r+1,0,0));
draw((0,0,0)--(0,0,h+1),dashed);

path3 w=(x1,0,s1)--(x2,0,s2)--(0,0,s2);
revolution b=revolution(w,Z);
draw(surface(b),blue+opacity(0.5),render);
draw(circle((0,0,s2),x2));
draw(circle((0,0,s1),x1));

draw("$x$",(xr,0,0)--(xr,0,sr),red,Arrow3,PenMargin3);
draw("$r$",(0,0,sr)--(xr,0,sr),N,red);
draw((string) r,(0,0,h)--(r,0,h),N,red);
draw((string) h,(r,0,0)--(r,0,h),red,Arrow3,PenMargin3);
draw((string) s,(-x,0,0)--(-x,0,s),W,red,Arrow3,Bar3,PenMargin3);
