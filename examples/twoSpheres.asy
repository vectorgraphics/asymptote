import three;

size(20cm);

currentprojection=orthographic(1,1,1);

draw(box((-2,-2,-1),(2,2,1)));

draw(shift(-Z)*surface(box((-2,-2),(2,2))),blue);
draw(shift(Z)*surface(box((-2,-2),(2,2))),orange+opacity(0.5));

surface s1=shift(0.5X+0.5Y)*unitsphere;
surface s2=shift(-0.5X-0.5Y)*unitsphere;

pen[] p=Gradient(green+opacity(0.6),white,green+opacity(0.6));

draw(s1,s1.palette(zpart,p));
draw(s2,s2.palette(zpart,p));
