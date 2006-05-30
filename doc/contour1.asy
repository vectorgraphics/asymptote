import contour;
size(200);
real f(real a, real b){return a^2+b^2;}
contour(f,(-1,-1),(1,1),new real[] {1});
