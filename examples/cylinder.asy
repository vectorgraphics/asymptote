import solids;

size(0,100);
currentlight=Viewport;

revolution r=cylinder(O,1,1.5,Y+Z);
draw(surface(r),green,render(merge=true));
draw(r,blue);
