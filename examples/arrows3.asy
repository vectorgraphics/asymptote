import three;

size(8cm);
currentprojection=orthographic(2,1,2,showtarget=false);
defaultpen(0.75mm);

path3 g=arc(O,1,90,0,90,110);

draw(g,blue,Arrow3(TeXHead2,blue));
draw(g,invisible,MidArrow3(TeXHead3,blue));
draw(scale3(3)*g,green,Arrow3(HookHead2));
draw(scale3(3)*g,invisible,MidArrow3(HookHead3,green));
draw(scale3(6)*g,red,Arrow3(DefaultHead2));
draw(scale3(6)*g,invisible,MidArrow3(DefaultHead3,red));
