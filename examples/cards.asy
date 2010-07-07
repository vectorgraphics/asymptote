picture rect;

size(rect,0,2.5cm);

real x=1;
real y=1.25;

filldraw(rect,box((-x,-y)/2,(x,y)/2),lightolive);

label(rect,"1",(-x,y)*0.45,SE);
label(rect,"2",(x,y)*0.45,SW);
label(rect,"3",(-x,-y)*0.45,NE);
label(rect,"4",(x,-y)*0.45,NW);

frame rectf=rect.fit();
frame toplef=rectf;
frame toprig=xscale(-1)*rectf;
frame botlef=yscale(-1)*rectf;
frame botrig=xscale(-1)*yscale(-1)*rectf;

size(0,7.5cm);

add(toplef,(-x,y));
add(toprig,(x,y));
add(botlef,(-x,-y));
add(botrig,(x,-y));
