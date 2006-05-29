import contour;
size(200);
real f(real x, real y){return x^2-y^2;}
int n=25;
real[] c = new real[n];
for(int i=0; i < n; ++i) c[i]=(i-n/2)/n;
contour(f,(-1,-1),(1,1),c,50,
	new pen(real c) {return c >= 0 ? currentpen : currentpen+dashed;});
