import feynman;

// draw a blue rectangle. This helps demonstrating how the draw commands
// erase the background by painting in white
fill((-50,-50)--(-50,100)--(50,100)--(50,-50)--cycle, blue);

// construct a path for the gluon
path p = (0,0){NW}..(0,100){right}..{SW}(0,0);

// draw a scalar line across the picture. At the intersections with the
// gluon line, it will be blotted out.
drawScalar((-50,0)--(50,100));

// place a label near the middle of the scalar line
label("$p$", midLabelPoint((-50,0)--(50,100), left), NW);

// put a small arrow next to the middle of the scalar line. Often used to
// indicate the direction of momentum. With the last parameter, we can shift
// it along the path.
drawReverseMomArrow((-50,0)--(50,100), left, 15);

// Yes, we can draw double lines
drawDoubleLine((-50,-50)--(0,0));

// and ghost lines (dotted)
drawGhost((0,0)--(50,-50));

// and finally the gluon line. For the size of the loops we use the default
// values. Here, we choose red as background colour. The last parameter
// affects the background erasion near the endpoints. Note that the other lines
// do not erase the background near the endpoints. This way, two lines can be
// joined without overdrawing each other
drawGluon(p,true,90);

// and a curved momentum arrow along the gluon line
drawMomArrow(p, right);

// a vertex where all the lines join
drawVertexOX((0,0));

shipout();

