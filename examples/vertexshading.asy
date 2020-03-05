import three;

size(200);

currentprojection=perspective(4,5,5);

draw(shift(2Z)*surface(O--X--Y--cycle,
                       new pen[] {red+opacity(0.5),green,blue}));
draw(shift(2Y+2Z)*surface(O--X--Y--cycle),blue);
draw(shift(2Y+Z)*surface(unitsquare3),green);

draw(surface(unitcircle3,new pen[] {red,green,blue,black}));
draw(surface(shift(Z)*unitsquare3,
             new pen[] {red,green+opacity(0.5),blue,black}),
     prc() ? nolight : currentlight);
draw(surface(shift(X)*((0,0,0)..controls (1,0,0) and (2,0,0)..(3,0,0)..
             controls (2.5,sqrt(3)/2,0) and (2,sqrt(3),0)..
             (1.5,3*sqrt(3)/2,0)..
                         controls (1,sqrt(3),0) and (0.5,sqrt(3)/2,0)..cycle),
             new triple[] {(1.5,sqrt(3)/2,2)},new pen[] {red,green,blue}));
