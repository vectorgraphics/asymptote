import obj;

size(15cm);
currentprojection=orthographic(0,2,5,up=Y);

// A compressed version of the required data file may be obtained from:
// http://www.cs.technion.ac.il/~irit/data/Viewpoint/galleon.obj.gz

pen[] surfacepen={darkred,brown,darkred+orange,heavyred,heavyred,darkred+orange,
                  palegreen+blue+lightgrey,darkred,darkred,yellow,darkred,white,
                  white,white,white,white,white};
surfacepen.cyclic=true;

draw(obj("galleon.obj",verbose=false,surfacepen));
