import graph3;

size(200,0);
currentprojection=orthographic(4,0,2.0);

real R=2.0;
real a=1.9;

triple f(pair t) {
  return ((R+a*cos(t.y))*cos(t.x),(R+a*cos(t.y))*sin(t.x),a*sin(t.y));
}

pen p=rgb(0.2,0.5,0.7);

// surface only
//add(surface(f,(0,0),(2pi,2pi),outward=true,30,15));

// mesh only
//add(surface(f,(0,0),(2pi,2pi),outward=true,30,15,nullpen,meshpen=p));

// surface & mesh
//add(surface(f,(0,0),(2pi,2pi),outward=true,30,15,meshpen=p));

// Surface coloring looks better if seam is moved
add(surface(f,(pi,pi),(3pi,3pi),outward=true,30,15,meshpen=p));
