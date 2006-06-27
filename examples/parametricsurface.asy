import graph3;

size(200,0);
currentprojection=orthographic((4,0,2.0));

triple f(pair t) {
    return ((2.0+1.9*cos(t.y))*cos(t.x),
    (2.0+1.9*cos(t.y))*sin(t.x),
    1.9*sin(t.y));
}

pen p=rgb(0.2,0.5,0.7);

// surface only
//add(surface(f,(0,0),(2pi,2pi),30,15));

// mesh only
//add(surface(f,(0,0),(2pi,2pi),30,15,nullpen,meshpen=p));

// surface & mesh
//add(surface(f,(0,0),(2pi,2pi),30,15,meshpen=p));

// Surface coloring looks better if seam is moved
add(surface(f,(pi,pi),(3pi,3pi),30,15,meshpen=p));
