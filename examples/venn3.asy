size(0,150);

pen colour1=red;
pen colour2=green;
pen colour3=blue;

real r=sqrt(3);

pair z0=(0,0);
pair z1=(-1,0);
pair z2=(1,0);
pair z3=(0,r);

path c1=circle(z1,r);
path c2=circle(z2,r);
path c3=circle(z3,r);

fill(c1,colour1);
fill(c2,colour2);
fill(c3,colour3);

picture intersection12;
fill(intersection12,c1,colour1+colour2);
clip(intersection12,c2);

picture intersection13;
fill(intersection13,c1,colour1+colour3);
clip(intersection13,c3);

picture intersection23;
fill(intersection23,c2,colour2+colour3);
clip(intersection23,c3);

picture intersection123;
fill(intersection123,c1,colour1+colour2+colour3);
clip(intersection123,c2);
clip(intersection123,c3);

add(intersection12);
add(intersection13);
add(intersection23);
add(intersection123);

draw(c1);
draw(c2);
draw(c3);

label("$A$",z1);
label("$B$",z2);
label("$C$",z3);
