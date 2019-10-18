import three;

size(15cm);

defaultrender.merge=true;

currentprojection=perspective(24,14,13);
currentlight=light(gray(0.5),specularfactor=3,
                   (0.5,-0.5,-0.25),(0.5,0.5,0.25),(0.5,0.5,1),(-0.5,-0.5,-1));

defaultpen(0.75mm);

path3 g=arc(O,1,90,-60,90,60);

draw(g,blue,Arrows3(TeXHead3),currentlight);
draw(scale3(3)*g,green,ArcArrows3(HookHead3),currentlight);
draw(scale3(6)*g,red,Arrows3(DefaultHead3),currentlight);

transform3 t=shift(invert(3S,O));

draw(t*g,blue,Arrows3(TeXHead2),currentlight);
draw(t*scale3(3)*g,green,ArcArrows3(HookHead2,NoFill),currentlight);
draw(t*scale3(6)*g,red,Arrows3(DefaultHead2(normal=Z)),currentlight);

transform3 t=shift(invert(6S,O));

draw(t*g,blue,Arrow3(TeXHead3,position=Relative(0.5)),currentlight);
draw(t*scale3(3)*g,purple,Arrow3(HookHead3,position=Relative(0.5)),
     currentlight);
draw(t*scale3(6)*g,red,Arrow3(DefaultHead3,position=Relative(0.5)),
     currentlight);

transform3 t=shift(invert(9S,O));

draw(t*g,blue,Arrow3(TeXHead2,position=Relative(0.5)),currentlight);
draw(t*scale3(3)*g,green,Arrow3(HookHead2,position=Relative(0.5),NoFill),
     currentlight);
draw(t*scale3(6)*g,red,Arrow3(DefaultHead2(normal=Z),position=Relative(0.5)),
     currentlight);
