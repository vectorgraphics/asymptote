import math;

size(100,0);

pair z4=(0,0);
pair z7=(2,0);
pair z1=point(rotate(60)*(z4--z7),1);

pair z5=interp(z4,z7,0.5);
pair z3=interp(z7,z1,0.5);
pair z2=interp(z1,z4,0.5);
pair z6=extension(z4,z3,z7,z2);

draw(z4--z7--z1--cycle);
draw(z4--z3);
draw(z7--z2);
draw(z1--z5);
draw(circle(z6,abs(z3-z6)));

label("1",z1,dir(z5--z1));
label("2",z2,dir(z7--z2));
label("3",z3,dir(z4--z3));
label("4",z4,dir(z3--z4));
label("5",z5,dir(z1--z5));
label("6",z6,2.5E+0.1*N);
label("7",z7,dir(z2--z7));


