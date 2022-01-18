import three;
import palette;

size(20cm);

currentprojection=orthographic(1,1,1);

draw(box((-2,-2,-1),(2,2,1)));

draw(shift(-Z)*surface(box((-2,-2),(2,2))),blue);
draw(shift(Z)*surface(box((-2,-2),(2,2))),orange+opacity(0.5));

surface s=unitsphere;
s.colors(palette(s.map(zpart),Gradient(green+opacity(0.6),white,
                                       green+opacity(0.6))));
draw(shift(0.5X+0.5Y)*s);
draw(shift(-0.5X-0.5Y)*s);
