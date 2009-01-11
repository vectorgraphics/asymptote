size(200);
import solids;
settings.render=0;
settings.prc=false;

currentprojection=perspective(4,4,3);
revolution hyperboloid=revolution(new real(real x) {return sqrt(1+x*x);},
                                  -2,2,20,operator..,X);
draw(hyperboloid.silhouette(64),blue);
