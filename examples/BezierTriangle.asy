import three;
currentprojection=perspective(-2,5,1);

size(10cm);

surface s=surface((0,0,0)..controls (1,0,0) and (2,0,0)..(3,0,0)..
                  controls (2.5,sqrt(3)/2,0) and (2,sqrt(3),0)..
                  (1.5,3*sqrt(3)/2,0)..
                  controls (1,sqrt(3),0) and (0.5,sqrt(3)/2,0)..cycle,
                  new triple[] {(1.5,sqrt(3)/2,2)});

draw(s,red);

