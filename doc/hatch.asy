size(0,100);
import patterns;

add("hatch",hatch());
add("hatchback",hatch(NW));
add("crosshatch",crosshatch());

filldraw(unitsquare,pattern("hatch"));
filldraw(shift(1,0)*unitsquare,pattern("hatchback"));
filldraw(shift(2,0)*unitsquare,pattern("crosshatch"));
shipout();
