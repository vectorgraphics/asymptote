import graph3;
import palette;

size3(200,IgnoreAspect);

file in=input("filesurface.dat").line();
real[] x=in;
real[] y=in;
real[][] z=in;

surface s=surface(z,x,y,linear,linear);
real[] level=uniform(min(z)*(1-sqrtEpsilon),max(z)*(1+sqrtEpsilon),4);

s.colors(palette(s.map(new real(triple v) {return find(level >= v.z);}),
                 Rainbow()));

draw(s,meshpen=thick(),render(merge=true));

triple m=currentpicture.userMin();
triple M=currentpicture.userMax();
triple target=0.5*(m+M);

xaxis3("$x$",Bounds,InTicks);
yaxis3("$y$",Bounds,InTicks(Step=1,step=0.1));
zaxis3("$z$",Bounds,InTicks);

/*
  picture palette;
  size3(palette,1cm);
  draw(palette,unitcube,red);
  frame F=palette.fit3();
  add(F,(M.x,m.y,m.z));
*/

currentprojection=perspective(camera=target+realmult(dir(68,225),M-m),
                              target=target);
