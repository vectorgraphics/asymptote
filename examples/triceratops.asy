import obj;

size(15cm);
currentprojection=orthographic(0,2,5,up=Y);

// A compressed version of the required data file may be obtained from:
// http://www.cs.technion.ac.il/~irit/data/Viewpoint/triceratops.obj.gz

draw(obj("triceratops.obj",brown));
