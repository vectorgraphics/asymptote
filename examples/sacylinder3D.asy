import solids;
size(0,100);

real r=1;
real h=3;

revolution R=cylinder(-h/2*Z,r,h);
draw(surface(R),lightgreen+opacity(0.5),render(compression=Low));
draw((0,0,-h/2)--(0,0,h/2),dashed);
dot((0,0,-h/2));
dot((0,0,h/2));
draw("$L$",(0,r,-h/2)--(0,r,h/2),W,black);
draw("$r$",(0,0,-h/2)--(0,r,-h/2),red);
draw(arc(O,1,90,90,90,0),red,Arrow3);
