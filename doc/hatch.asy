size(100,0);

picture tiling=new picture;
draw(tiling,scale(5mm)*unitsquare,linecap(Square)+linejoin(Miter));
add("hatch",tiling,min(currentpen),max(currentpen));

fill(unitcircle,pattern("hatch"));
shipout();
