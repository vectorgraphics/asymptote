import three;
import graph;
import graph3;

size(0,175);

currentprojection=orthographic(500,-500,500);

triple[] z=new triple[10];

z[0]=(0,100,0); z[1]=(50,0,0); z[2]=(180,0,0);

for(int n=3; n <= 9; ++n)
  z[n]=z[n-3]+(200,0,0);

path3 p=z[0]..z[1]---z[2]::{Y}z[3]
     &z[3]..z[4]--z[5]::{Y}z[6]
     &z[6]::z[7]---z[8]..{Y}z[9];

draw(p,grey+linewidth(4mm));

bbox3 b=limits(O,(700,200,100));

xaxis(Label("$x$",1),b,red,Arrow);
yaxis(Label("$y$",1),b,red,Arrow);
zaxis(Label("$z$",1),b,red,Arrow);

dot(z);

