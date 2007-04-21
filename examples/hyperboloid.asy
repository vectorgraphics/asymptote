size(200);
import solids;

currentprojection=perspective(4,4,3);
revolution hyperboloid=revolution(new real(real x) {return sqrt(1+x*x);},
                                  -2,2,12,X);   
hyperboloid.filldraw(green,6,blue,false);

