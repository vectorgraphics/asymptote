import slopefield;

size(200);

real func(real x) {return 2x;}
add(slopefield(func,(-3,-3),(3,3),20,Arrow));
draw(curve((0,0),func,(-3,-3),(3,3)),red);


