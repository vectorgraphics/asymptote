size(0,150);

pen colour1=red;
pen colour2=green;

pair z0=(0,0);
pair z1=(-1,0);
pair z2=(1,0);
real r=1.5;
guide c1=circle(z1,r);
guide c2=circle(z2,r);
fill(c1,colour1);
fill(c2,colour2);

picture intersection=new picture;
fill(intersection,c1,colour1+colour2);
clip(intersection,c2);

add(intersection);

draw(c1);
draw(c2);

label("$A$",z1);
label("$B$",z2);

pair z=(0,-2);
real e=0.85;
draw("$A\cap B$",subpath(conj(z)--z0,0,e),0,Arrow);
draw("$A\cup B$",subpath(z--z0,0,e),0,Arrow);
draw(subpath(z--z1,0,e),Arrow);
draw(subpath(z--z2,0,e),Arrow);

shipout(bbox(0.25cm));
