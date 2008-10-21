import obj;

size(15cm);
currentprojection=orthographic(4,2,5,up=Y);

// Compressed data may be obtained from:
// http://www.cs.technion.ac.il/~irit/data/Viewpoint/galleon.obj.gz

pen[] surfacepen={darkred,brown,darkred+orange,heavyred,heavyred,darkred+orange,
                  palegreen+blue+lightgrey,darkred,darkred,yellow,darkred,white,
                  white,white,white,white,white};

pen[] meshpen={darkgrey};
surfacepen.cyclic(true);
meshpen.cyclic(true);

draw(obj("galleon.obj",true,surfacepen,meshpen));
