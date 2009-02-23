import three;

size(8cm);
currentprojection=orthographic(1,1,2,showtarget=false);
defaultpen(0.75mm);

path3 g=arc(O,1,90,-60,90,90);

draw(g,blue,Arrows3(TeXHead3),currentlight);
draw(g,invisible,MidArrow3(TeXHead2,Fill(blue)));
draw(scale3(3)*g,green,ArcArrows3(HookHead3,NoFill),currentlight);
draw(scale3(3)*g,invisible,MidArcArrow3(HookHead2,Fill(green)));
draw(scale3(6)*g,red,Arrows3(DefaultHead2,FillDraw(red)),currentlight);
draw(scale3(6)*g,invisible,MidArrow3(DefaultHead3,Fill(red)));

