import graph3;

// Beginnings of a solid geometry package.

// Return a cylinder of height h constructed from area base in the XY plane
// and aligned with axis.
guide3[] cylinder(guide3 base, real h, triple axis=Z,
		  projection P=currentprojection) 
{
  base=rotate(-colatitude(axis),cross(axis,Z))*base;
  guide3 top=shift(h*axis)*base;
  
  path Base=base;
  path Top=top;
  if((inside(Base,point(Top,0)) || inside(Top,point(Base,0))) &&
     (abs(dot(axis,P.camera)) > abs(axis)*abs(P.camera)*(1-epsilon) || 
      intersect(Base,Top) == (-1,-1)))
    return base^^top;
     
  triple c=0.5*(min(base)+max(base)); 
 
  // Iterate to determine the cylinder edges, accounting for perspective
  triple intersectionpoint(guide3 face, real h, bool left, int n=30) {
    triple a=O;
    for(int i=0; i < n; ++i) {
      pair z=project(a+axis,P)-project(a,P);
      real angle=degrees(z,warn=false)-90;
      path g=rotate(-angle)*face;
      pair M=max(g);
      pair m=min(g);
      a=invert(rotate(angle)*
	       intersectionpoint(left ? (m.x,m.y)--(m.x,M.y) :
				 (M.x,m.y)--(M.x,M.y),g),axis,c+h*axis);
    }
    return a;
  }

  return base^^intersectionpoint(base,0,true)--intersectionpoint(top,h,true)^^
    intersectionpoint(base,0,false)--intersectionpoint(top,h,false)^^top;
}

// Return a cylinder of height h constructed from circle(c,r,Z)
// and aligned with axis.
guide3[] cylinder(triple c, real r, real h, triple axis=Z,
		  projection P=currentprojection) {
  return cylinder(circle(c,r,Z),h,axis);
}
