size(0,100);
import patterns;

real d=4mm;
picture tiling=new picture;
guide square=scale(d)*unitsquare;
fill(tiling,square,white,(0,0),black,(d,d));
fill(tiling,shift(d,d)*square,blue);
add("checked",tiling);

filldraw(unitcircle,pattern("checked"));
shipout();

