import three;

size(200);

currentprojection=perspective(4,5,5);

draw(surface(unitcircle3,new pen[] {red,green,blue,white}));
draw(surface(shift(Z)*unitsquare3,
	     new pen[] {red,green+opacity(0.5),blue,black}));
