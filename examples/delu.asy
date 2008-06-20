size(7cm,0);

pair z1=(1,-0.25);
pair v1=dir(45);
pair z2=-z1;
pair v2=0.75*dir(260);
pair z3=(z1.x,-3);

// A centered random number
real crand() {return unitrand()-0.5;}

guide g;
pair lastz;
for(int i=0; i < 60; ++i) {
  pair z=0.75*lastz+(crand(),crand());
  g=g..2.5*z;
  lastz=z;
}
g=shift(0,-.5)*g..cycle;

draw(g,gray(0.7));

draw("$r$",z1--z2,RightSide,red,Arrows,DotMargins);
draw(z1--z1+v1,Arrow);
draw(z2--z2+v2,Arrow);
draw(z3--z3+v1-v2,green,Arrow);

dot("1",z1,S,blue);
dot("2",z2,NW,blue);
