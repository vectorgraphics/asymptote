import three;
import palette; 
import graph3;

size(300);

currentprojection=perspective(-30,-30,30,up=Z);

surface s;

for(int i = 0; i < 10; ++i) {
  for(int j = 0; j < 10; ++j) {
    s.append(shift(i,j,0)*scale(1,1,i+j)*unitcube);
  }
}

s.colors(palette(s.map(zpart),Rainbow()));
draw(s,meshpen=black+thick(),nolight,render(merge=true));

xaxis3("$x$",Bounds,InTicks(endlabel=false,Label,2,2));
yaxis3(YZ()*"$y$",Bounds,InTicks(beginlabel=false,Label,2,2));
zaxis3(XZ()*"$z$",Bounds,InTicks);
