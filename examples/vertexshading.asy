import three;
import graph3;

size(200);

currentprojection=perspective(4,5,5);

draw(shift(2Z)*surface(O--X--Y--cycle),cornerPen(red+opacity(0.5),green,blue));
draw(shift(2Y+2Z)*surface(O--X--Y--cycle),blue);
draw(shift(2Y+Z)*surface(unitsquare3),green);

draw(surface(unitcircle3),cornerPen(red,green,blue,black));
draw(surface(shift(Z)*unitsquare3),
     cornerPen(red,green+opacity(0.5),blue,black),
     prc() ? nolight : currentlight);
draw(surface(shift(X)*((0,0,0)..controls (1,0,0) and (2,0,0)..(3,0,0)..
                       controls (2.5,sqrt(3)/2,0) and (2,sqrt(3),0)..
                       (1.5,3*sqrt(3)/2,0)..
                       controls (1,sqrt(3),0) and (0.5,sqrt(3)/2,0)..cycle),
             new triple[] {(1.5,sqrt(3)/2,2)}),cornerPen(red,green,blue));

surface S=surface(new real(pair z) {return 1;},(0,0),(1,1),2,2);

draw(shift(X+Y)*S,cornerPen(new pen[][] {
      {red,green,blue,black},
        {black,blue,green,red},
        {green,red,black,blue},
        {blue,black,red,green},
        }));
