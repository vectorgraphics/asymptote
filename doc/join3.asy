import graph3;

size(200);

currentprojection=orthographic(500,-500,500);

triple[] z=new triple[10];

z[0]=(0,100,0); z[1]=(50,0,0); z[2]=(180,0,0);

for(int n=3; n <= 9; ++n)
  z[n]=z[n-3]+(200,0,0);

path3 p=z[0]..z[1]---z[2]::{Y}z[3]
&z[3]..z[4]--z[5]::{Y}z[6]
&z[6]::z[7]---z[8]..{Y}z[9];

draw(p,grey+linewidth(4mm),currentlight);

xaxis3(Label(XY()*"$x$",align=-3Y),red,above=true);
yaxis3(Label(XY()*"$y$",align=-3X),red,above=true);
