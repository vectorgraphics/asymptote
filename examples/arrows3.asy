import three;

size(8cm);
currentprojection=orthographic(4,4,5,showtarget=false);
defaultpen(0.75mm);

path3 g=arc(O,1,90,-45,90,90);

draw(g,blue,Arrow3(TeXHead3,blue));
draw(g,invisible,MidArrow3(TeXHead2,blue));
draw(scale3(3)*g,green,Arrow3(HookHead3));
draw(scale3(3)*g,invisible,MidArrow3(HookHead2,green));
draw(scale3(6)*g,red,Arrow3(DefaultHead3));
draw(scale3(6)*g,invisible,MidArrow3(DefaultHead2,red));
