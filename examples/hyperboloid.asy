size(200);
import solids;

currentprojection=perspective(4,4,3);
revolution hyperboloid=revolution(new real(real x) {return sqrt(1+x*x);},
                                  -2,2,20,operator..,X);
draw(surface(hyperboloid),green,render(compression=Low,merge=true));
draw(hyperboloid,6,blue,longitudinalpen=nullpen);
