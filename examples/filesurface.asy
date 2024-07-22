import graph3;
import palette;

size3(200,IgnoreAspect);

file in=input("filesurface.dat").line();
real[] x=in;
real[] y=in;
real[][] z=in;

surface s=surface(z,x,y);
real[] level=uniform(min(z)*(1-sqrtEpsilon),max(z)*(1+sqrtEpsilon),256);

s.colors(palette(s.map(new real(triple v) {return find(level >= v.z);}),
                 Rainbow()));

draw(s,meshpen=thick(),render(tessellate=true));

triple m=currentpicture.userMin();
triple M=currentpicture.userMax();
triple target=0.5*(m+M);

xaxis3("$x$",Bounds,InTicks);
yaxis3("$y$",Bounds,InTicks(Step=1,step=0.1));
zaxis3("$z$",Bounds,InTicks);

currentprojection=perspective(camera=target+realmult(dir(68,225),M-m),
                              target=target);
