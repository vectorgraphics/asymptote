size(0,100);
import patterns;

add("hatch",hatch());
add("hatchback",hatch(NW));
add("crosshatch",crosshatch(3mm));
add("brick",brick());

int c=0;
filldraw(shift(++c,0)*unitsquare,pattern("hatch"));
filldraw(shift(++c,0)*unitsquare,pattern("hatchback"));
filldraw(shift(++c,0)*unitsquare,pattern("crosshatch"));
filldraw(shift(++c,0)*unitsquare,pattern("brick"));
shipout();
